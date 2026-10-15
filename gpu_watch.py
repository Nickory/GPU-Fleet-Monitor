#!/usr/bin/env python3
"""Multi-server GPU monitor for a local terminal."""

from __future__ import annotations

import argparse
import curses
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time
import textwrap
from typing import Any



DEFAULT_SERVERS = [
    {"name": "GPU Server 01", "connection": "gpu01.example.invalid:22"},
    {"name": "GPU Server 02", "connection": "gpu02.example.invalid:22"},
    {"name": "GPU Server 03", "connection": "gpu03.example.invalid:22"},
]

REMOTE_SCRIPT = r'''
import csv
import io
import json
import os
import pwd
import subprocess

def value(obj, name):
    try:
        item = getattr(obj, name)
        return item() if callable(item) else item
    except Exception:
        return None

def clean(item):
    if item is None:
        return None
    if isinstance(item, str) and item.strip().lower() in ("n/a", "na", ""):
        return None
    if isinstance(item, (str, int, float, bool)):
        return item
    try:
        return int(item)
    except Exception:
        return str(item)

def number(obj, name):
    item = clean(value(obj, name))
    if item is None:
        return None
    try:
        return float(item)
    except (ValueError, TypeError):
        return None

def run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "command failed: " + command[0])
    return result.stdout

def collect_nvitop():
    from nvitop import Device
    gpus = []
    for device in Device.all():
        try:
            raw_processes = value(device, "processes") or {}
            processes = []
            if hasattr(raw_processes, "values"):
                for process in raw_processes.values():
                    command = clean(value(process, "command"))
                    processes.append({
                        "pid": clean(value(process, "pid")),
                        "user": clean(value(process, "username")),
                        "command": str(command)[:500] if command is not None else None,
                        "gpu_memory_bytes": number(process, "gpu_memory"),
                        "gpu_memory_human": clean(value(process, "gpu_memory_human")),
                        "running_time": clean(value(process, "running_time_human")),
                        "gpu_sm_utilization": number(process, "gpu_sm_utilization"),
                        "gpu_memory_utilization": number(process, "gpu_memory_utilization"),
                        "gpu_context_type": clean(value(process, "type")),
                        "cpu_percent": number(process, "cpu_percent"),
                        "host_memory_bytes": number(process, "host_memory"),
                        "host_memory_human": clean(value(process, "host_memory_human")),
                        "host_memory_percent": number(process, "host_memory_percent") if number(process, "host_memory_percent") is not None else number(process, "memory_percent"),
                    })
            processes.sort(key=lambda item: (item["gpu_memory_bytes"] or 0), reverse=True)
            clocks = value(device, "clock_infos")
            gpus.append({
                "index": clean(value(device, "index")),
                "name": clean(value(device, "name")),
                "uuid": clean(value(device, "uuid")),
                "bus_id": clean(value(device, "bus_id")),
                "gpu_utilization": number(device, "gpu_utilization"),
                "memory_utilization": number(device, "memory_utilization"),
                "memory_total_bytes": number(device, "memory_total"),
                "memory_used_bytes": number(device, "memory_used"),
                "memory_percent": number(device, "memory_percent"),
                "temperature": number(device, "temperature"),
                "fan_speed": number(device, "fan_speed"),
                "power_usage_w": number(device, "power_usage"),
                "power_limit_w": number(device, "power_limit"),
                "clock_graphics_mhz": number(clocks, "graphics") if clocks is not None else None,
                "clock_memory_mhz": number(clocks, "memory") if clocks is not None else None,
                "processes": processes,
            })
        except Exception as error:
            gpus.append({"index": clean(value(device, "index")), "error": str(error)})
    try:
        driver_version = clean(Device.driver_version())
    except Exception:
        driver_version = None
    return {"source": "nvitop", "driver_version": driver_version, "gpus": gpus}

def numeric_field(row, index):
    try:
        item = row[index].strip()
        if item.lower() in ("n/a", "not supported", ""):
            return None
        return float(item)
    except (IndexError, ValueError, TypeError):
        return None

def collect_nvidia_smi():
    gpu_text = run([
        "nvidia-smi",
        "--query-gpu=driver_version,index,name,uuid,pci.bus_id,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,fan.speed,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ])
    gpu_rows = list(csv.reader(io.StringIO(gpu_text)))
    gpus = []
    by_uuid = {}
    driver_version = None
    for row in gpu_rows:
        if len(row) < 13:
            continue
        driver_version = driver_version or row[0].strip()
        uuid = row[3].strip()
        total_mib = numeric_field(row, 5)
        used_mib = numeric_field(row, 6)
        gpus.append({
            "index": numeric_field(row, 1),
            "name": row[2].strip(),
            "uuid": uuid,
            "bus_id": row[4].strip() or None,
            "gpu_utilization": numeric_field(row, 7),
            "memory_utilization": numeric_field(row, 8),
            "memory_total_bytes": total_mib * 1024 * 1024 if total_mib is not None else None,
            "memory_used_bytes": used_mib * 1024 * 1024 if used_mib is not None else None,
            "temperature": numeric_field(row, 9),
            "fan_speed": numeric_field(row, 10),
            "power_usage_w": numeric_field(row, 11),
            "power_limit_w": numeric_field(row, 12),
            "processes": [],
        })
        by_uuid[uuid] = gpus[-1]
    try:
        process_text = run([
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
    except Exception:
        process_text = ""
    for row in csv.reader(io.StringIO(process_text)):
        if len(row) < 4:
            continue
        gpu = by_uuid.get(row[0].strip())
        if gpu is None:
            continue
        memory_mib = numeric_field(row, 3)
        process = {
            "pid": numeric_field(row, 1),
            "user": None,
            "command": row[2].strip(),
            "gpu_memory_bytes": memory_mib * 1024 * 1024 if memory_mib is not None else None,
            "gpu_memory_human": str(round(memory_mib)) + " MiB" if memory_mib is not None else None,
        }
        gpu["processes"].append(process)
    try:
        pmon_text = run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"])
    except Exception:
        pmon_text = ""
    for line in pmon_text.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[0].startswith("#"):
            continue
        try:
            gpu_index = int(fields[0])
            pid = int(fields[1])
        except ValueError:
            continue
        gpu = next((item for item in gpus if item.get("index") is not None and int(float(item["index"])) == gpu_index), None)
        if gpu is None:
            continue
        process = next((item for item in gpu["processes"] if int(float(item.get("pid") or -1)) == pid), None)
        if process is None:
            continue
        process["gpu_context_type"] = fields[2] if fields[2] not in ("-", "N/A") else None
        process["gpu_sm_utilization"] = numeric_field(fields, 3)
        process["gpu_memory_utilization"] = numeric_field(fields, 4)
    return {"source": "nvidia-smi", "driver_version": driver_version, "gpus": gpus}

def attach_owner_identity(result):
    current_uid = os.geteuid()
    try:
        current_user = pwd.getpwuid(current_uid).pw_name
    except KeyError:
        current_user = str(current_uid)
    process_records = [
        process
        for gpu in result.get("gpus", [])
        for process in gpu.get("processes", [])
    ]
    pids = set()
    for process in process_records:
        try:
            pids.add(str(int(float(process.get("pid")))))
        except (TypeError, ValueError):
            pass
    owners = {}
    if pids:
        try:
            output = subprocess.run(
                ["ps", "-ww", "-o", "user=,uid=,pid=,ppid=,pcpu=,pmem=,rss=,etime=,comm=,args=", "-p", ",".join(sorted(pids))],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            ).stdout
            for line in output.splitlines():
                fields = line.split(None, 9)
                if len(fields) >= 9:
                    try:
                        uid = int(fields[1])
                        ppid = int(fields[3])
                        rss_kib = numeric_field(fields, 6)
                        owners[fields[2]] = {
                            "user": fields[0],
                            "uid": uid,
                            "ppid": ppid,
                            "cpu_percent": numeric_field(fields, 4),
                            "host_memory_percent": numeric_field(fields, 5),
                            "host_memory_bytes": rss_kib * 1024 if rss_kib is not None else None,
                            "running_time": fields[7],
                            "process_name": fields[8],
                            "command": fields[9] if len(fields) >= 10 else fields[8],
                        }
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass
    for process in process_records:
        try:
            pid = str(int(float(process.get("pid"))))
        except (TypeError, ValueError):
            continue
        owner = owners.get(pid)
        if owner is not None:
            for field, item in owner.items():
                if process.get(field) is None:
                    process[field] = item
            if owner.get("command") and len(owner["command"]) > len(str(process.get("command") or "")):
                process["command"] = owner["command"][:500]
            if process.get("host_memory_human") is None:
                process["host_memory_human"] = (
                str(round(owner["host_memory_bytes"] / (1024 * 1024))) + " MiB"
                if owner.get("host_memory_bytes") is not None else None
                )
    result["remote_user"] = current_user
    result["remote_uid"] = current_uid

try:
    result = collect_nvitop()
except Exception as nvitop_error:
    try:
        result = collect_nvidia_smi()
    except Exception as nvidia_error:
        raise RuntimeError("nvitop: " + str(nvitop_error) + "; nvidia-smi: " + str(nvidia_error))

attach_owner_identity(result)

print("GPU_DASHBOARD_JSON=" + json.dumps(result, ensure_ascii=False))
'''

HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,252}$")
USER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9._-]{0,63}$")


def parse_connection(raw: str) -> tuple[str | None, str, int | None]:
    raw = raw.strip()
    if not raw or len(raw) > 512:
        raise ValueError("连接信息为空或过长。")

    user: str | None = None
    host: str | None = None
    port: int | None = None

    if raw.startswith("ssh "):
        try:
            tokens = shlex.split(raw)
        except ValueError as error:
            raise ValueError("SSH 命令格式无效。") from error
        if not tokens or tokens[0] != "ssh":
            raise ValueError("请填写 ssh 命令或 host:port。")
        target: str | None = None
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in ("-p", "-l"):
                if index + 1 >= len(tokens):
                    raise ValueError("SSH 参数缺少值。")
                argument = tokens[index + 1]
                if token == "-p":
                    if not argument.isdigit():
                        raise ValueError("SSH 端口必须是数字。")
                    port = int(argument)
                else:
                    user = argument
                index += 2
            elif token.startswith("-"):
                raise ValueError("目前支持 ssh 命令中的 -p 和 -l 参数；密钥请使用本机 SSH 配置或 ssh-agent。")
            elif target is None:
                target = token
                index += 1
            else:
                raise ValueError("请只填写 SSH 连接目标，不要附加远端命令。")
        if target is None:
            raise ValueError("SSH 命令中没有主机地址。")
        if "@" in target:
            if target.count("@") != 1 or user is not None:
                raise ValueError("SSH 用户名格式无效。")
            user, target = target.split("@", 1)
        if ":" in target:
            maybe_host, maybe_port = target.rsplit(":", 1)
            if maybe_port.isdigit():
                if port is not None and port != int(maybe_port):
                    raise ValueError("端口参数重复且不一致。")
                port = int(maybe_port)
                target = maybe_host
        host = target
    else:
        if any(character.isspace() for character in raw):
            raise ValueError("请填写 ssh 命令、user@host:port 或 host:port。")
        target = raw
        if "@" in target:
            if target.count("@") != 1:
                raise ValueError("SSH 用户名格式无效。")
            user, target = target.split("@", 1)
        if ":" in target:
            maybe_host, maybe_port = target.rsplit(":", 1)
            if not maybe_port.isdigit():
                raise ValueError("host:port 中的端口必须是数字。")
            port = int(maybe_port)
            target = maybe_host
        host = target

    if not host or not HOST_RE.fullmatch(host):
        raise ValueError("主机地址格式无效。")
    if user is not None and not USER_RE.fullmatch(user):
        raise ValueError("用户名格式无效。")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("端口必须在 1 到 65535 之间。")
    return user, host, port


def ssh_control_path() -> str | None:
    """Return a private, short socket path for reusing SSH sessions."""
    try:
        # Unix-domain socket paths have a small platform limit. Keep this in
        # a per-user private /tmp directory instead of the longer home path.
        directory = Path("/tmp") / f"gfm-{os.getuid()}"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        return str(directory / "%C")
    except OSError:
        # Multiplexing is an optimization; SSH still works without this cache.
        return None


def is_retryable_ssh_error(message: str) -> bool:
    lowered = message.casefold()
    return any(
        marker in lowered
        for marker in (
            "connection closed",
            "connection reset",
            "connection refused",
            "connection timed out",
            "operation timed out",
            "broken pipe",
            "kex_exchange_identification",
            "mux_client_request_session",
            "master is dead",
        )
    )


def query_server(item: dict[str, Any]) -> dict[str, Any]:
    name = str(item.get("name", "")).strip()[:80] or "未命名服务器"
    raw = str(item.get("connection", "")).strip()
    started = time.monotonic()
    try:
        user, host, port = parse_connection(raw)
        target = (user + "@" if user else "") + host
        address = target + (":" + str(port) if port else "")
        base_command = [
            "ssh",
            "-T",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=12",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            target,
        ]
        if port is not None:
            base_command[1:1] = ["-p", str(port)]
        remote_command = "python3 -c " + shlex.quote(REMOTE_SCRIPT)
        base_command.append(remote_command)
        command = base_command.copy()
        control_path = ssh_control_path()
        if control_path:
            command[1:1] = [
                "-o", "ControlMaster=auto",
                "-o", "ControlPersist=45",
                "-o", "ControlPath=" + control_path,
                "-o", "StreamLocalBindUnlink=yes",
            ]
        marker = "GPU_DASHBOARD_JSON="
        result = None
        payload = None
        for attempt in range(2):
            result = subprocess.run(
                command if attempt == 0 else base_command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=45,
                check=False,
            )
            for line in reversed(result.stdout.splitlines()):
                if line.startswith(marker):
                    payload = json.loads(line[len(marker):])
                    break
            if payload is not None or attempt == 1:
                break
            message = result.stderr.strip() or result.stdout.strip()
            if not is_retryable_ssh_error(message):
                break
            time.sleep(0.6)
        elapsed = round((time.monotonic() - started) * 1000)
        if payload is None:
            assert result is not None
            message = result.stderr.strip() or result.stdout.strip() or ("SSH 退出码 " + str(result.returncode))
            return {
                "name": name, "connection": raw, "address": address, "ok": False,
                "error": message[-1200:], "elapsed_ms": elapsed,
            }
        return {
            "name": name, "connection": raw, "address": address, "ok": result.returncode == 0,
            "error": None if result.returncode == 0 else (result.stderr.strip()[-1200:] or "SSH 退出码 " + str(result.returncode)),
            "driver_version": payload.get("driver_version"),
            "source": payload.get("source"),
            "remote_user": payload.get("remote_user"),
            "remote_uid": payload.get("remote_uid"),
            "gpus": payload.get("gpus", []),
            "elapsed_ms": elapsed,
            "ssh_retries": attempt,
        }
    except subprocess.TimeoutExpired:
        error = "连接或 nvitop 查询超时（45 秒）。"
    except FileNotFoundError:
        error = "找不到 ssh 命令，请确认系统已安装 OpenSSH。"
    except (ValueError, json.JSONDecodeError) as exception:
        error = str(exception)
    except Exception as exception:
        error = str(exception)
    return {
        "name": name, "connection": raw, "ok": False,
        "error": error, "elapsed_ms": round((time.monotonic() - started) * 1000),
    }


def parse_servers(values: list[str] | None) -> list[dict[str, str]]:
    if not values:
        return [dict(item) for item in DEFAULT_SERVERS]
    servers: list[dict[str, str]] = []
    for value in values:
        label, separator, connection = value.partition("=")
        if separator and label.strip() and not value.lstrip().startswith("ssh "):
            name = label.strip()
            connection = connection.strip()
        else:
            name = value.strip()
            connection = value.strip()
        if not connection:
            raise ValueError("服务器连接信息不能为空。")
        servers.append({"name": name[:80], "connection": connection[:512]})
    return servers


def number(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def fmt_index(value: Any) -> str:
    parsed = number(value)
    if parsed is None:
        return str(value if value is not None else "?")
    return str(int(parsed)) if parsed.is_integer() else str(parsed)


def fmt_percent(value: Any) -> str:
    parsed = number(value)
    return "—" if parsed is None else f"{parsed:.0f}%"


def fmt_bytes(value: Any) -> str:
    parsed = number(value)
    if parsed is None:
        return "—"
    gib = parsed / (1024**3)
    return f"{gib:.1f}G" if gib >= 1 else f"{parsed / (1024**2):.0f}M"


def compact_runtime(value: Any) -> str:
    raw = str(value or "")
    if not raw:
        return "—"
    if "-" in raw:
        days, clock = raw.split("-", 1)
        hours = clock.split(":", 1)[0]
        return f"{days}d{hours}h"[:8]
    parts = raw.split(":")
    if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) >= 100:
        return f"{int(parts[0])}h{parts[1]}m"[:8]
    return raw[:8]


def progress(value: Any, width: int) -> str:
    parsed = number(value)
    if parsed is None:
        return " " * width
    ratio = max(0.0, min(100.0, parsed)) / 100.0
    filled = round(ratio * width)
    return "█" * filled + "·" * (width - filled)


def gpu_memory_percent(gpu: dict[str, Any]) -> float | None:
    used = number(gpu.get("memory_used_bytes"))
    total = number(gpu.get("memory_total_bytes"))
    if used is None or total is None or total <= 0:
        return number(gpu.get("memory_percent"))
    return used / total * 100.0


def loading_intensity(value: Any, moderate_threshold: int, heavy_threshold: int) -> int:
    parsed = number(value)
    if parsed is None:
        return 1  # nvitop treats unavailable utilization as moderate.
    if parsed >= heavy_threshold:
        return 2
    if parsed >= moderate_threshold:
        return 1
    return 0


def color_for_intensity(intensity: int, colors: dict[str, int]) -> int:
    return (colors["green"], colors["yellow"], colors["red"])[intensity]


def color_for_util(value: Any, colors: dict[str, int]) -> int:
    """Match nvitop's default GPU thresholds: 10% moderate, 75% heavy."""
    return color_for_intensity(loading_intensity(value, 10, 75), colors)


def color_for_memory(value: Any, colors: dict[str, int]) -> int:
    """Match nvitop's default VRAM thresholds: 10% moderate, 80% heavy."""
    return color_for_intensity(loading_intensity(value, 10, 80), colors)


def color_for_device_row(gpu_utilization: Any, memory_percent: Any, colors: dict[str, int]) -> int:
    """Color the device row using nvitop's maximum of GPU and VRAM intensity."""
    intensity = max(
        loading_intensity(gpu_utilization, 10, 75),
        loading_intensity(memory_percent, 10, 80),
    )
    return color_for_intensity(intensity, colors)


def build_content(
    servers: list[dict[str, Any]],
    width: int,
    colors: dict[str, int],
) -> tuple[list[tuple[bool, list[list[tuple[str, int, int]]]]], list[dict[str, Any]]]:
    """Build the GPU inventory and one combined, bottom-pane process list."""
    # Each item is (is_gpu_panel, visual_lines). The renderer shares horizontal
    # borders between adjacent GPU rows to keep the inventory dense.
    lines: list[tuple[bool, list[list[tuple[str, int, int]]]]] = []
    process_rows: list[dict[str, Any]] = []
    compact = width < 104
    bar_width = 2 if compact else (8 if width < 132 else 10)
    model_width = max(8, min(24, width - (68 if compact else 80)))

    def line(*segments: tuple[str, int, int]) -> None:
        lines.append((False, [list(segments)]))

    def fit_segments(segments: list[tuple[str, int, int]], target_width: int) -> list[tuple[str, int, int]]:
        fitted: list[tuple[str, int, int]] = []
        remaining = max(0, target_width)
        for text, color, attributes in segments:
            if remaining <= 0:
                break
            clipped = text[:remaining]
            if clipped:
                fitted.append((clipped, color, attributes))
                remaining -= len(clipped)
        if remaining > 0:
            fitted.append((" " * remaining, colors["muted"], 0))
        return fitted

    def gpu_panel(*segments: tuple[str, int, int]) -> None:
        frame_width = max(3, width - 1)
        inner_width = frame_width - 2
        body = [
            ("│", colors["muted"], 0),
            *fit_segments(list(segments), inner_width),
            ("│", colors["muted"], 0),
        ]
        lines.append((True, [body]))

    def bar_segments(value: Any, color: int) -> tuple[tuple[str, int, int], ...]:
        parsed = number(value)
        if parsed is None:
            return (
                ("[", colors["muted"], 0),
                ("·" * bar_width + "]", color, 0),
            )
        filled = round(max(0.0, min(100.0, parsed)) / 100.0 * bar_width)
        return (
            ("[", colors["muted"], 0),
            ("█" * filled, color, curses.A_BOLD),
            ("·" * (bar_width - filled) + "]", colors["muted"], 0),
        )

    # One shared heading keeps the per-host groups short and easy to scan.
    if compact:
        heading = f" GPU  {'MODEL':<{model_width}}  GPU% LOAD  VRAM  MEM  TEMP PROC"
    elif width < 132:
        heading = f" GPU  {'MODEL':<{model_width}}  GPU%  LOAD       VRAM            MEM        TEMP  POWER  PROC"
    elif width < 150:
        heading = f" GPU  {'MODEL':<{model_width}}  GPU%  LOAD         VRAM             MEM          TEMP   FAN       POWER   PROC"
    else:
        heading = f" GPU  {'MODEL':<{model_width}}  GPU%  LOAD         VRAM             MEM          TEMP   FAN       POWER   CLK G/M    PROC"
    line((heading, colors["muted"], curses.A_BOLD))

    if not servers:
        line((" No servers configured.", colors["muted"], 0))
        return lines, process_rows

    for server_index, server in enumerate(servers):
        name = server.get("name") or "GPU server"
        address = server.get("address") or server.get("connection") or ""
        if server.get("ok"):
            source = server.get("source", "nvitop")
            driver = server.get("driver_version")
            info = f"{source}"
            if driver:
                info += f" · drv {driver}"
            if server.get("elapsed_ms") is not None:
                info += f" · {server['elapsed_ms']}ms"
            if server.get("ssh_retries"):
                info += " · SSH retry"
            refresh_failures = int(server.get("_refresh_failures", 0) or 0)
            line(
                ("● ", colors["yellow"] if refresh_failures else colors["green"], curses.A_BOLD),
                (str(name), colors["cyan"], curses.A_BOLD),
                (f"  RETRY {refresh_failures}/3  " if refresh_failures else "  ", colors["yellow"], curses.A_BOLD if refresh_failures else 0),
                (f"  {address}  ", colors["white"], 0),
                (info, colors["muted"], 0),
            )
            server_gpus = server.get("gpus", [])
            if not server_gpus:
                line(("   No NVIDIA GPUs detected.", colors["muted"], 0))
            for gpu in server_gpus:
                if gpu.get("error"):
                    line((f"   GPU {fmt_index(gpu.get('index'))}: {gpu['error']}", colors["red"], 0))
                    continue
                index = fmt_index(gpu.get("index"))
                model = str(gpu.get("name") or "NVIDIA GPU")
                model = model[:model_width]
                model_cell = f"{model:<{model_width}}"
                gpu_util = number(gpu.get("gpu_utilization"))
                mem_pct = gpu_memory_percent(gpu)
                row_color = color_for_device_row(gpu_util, mem_pct, colors)
                gpu_color = color_for_util(gpu_util, colors)
                mem_color = color_for_memory(mem_pct, colors)
                used = fmt_bytes(gpu.get("memory_used_bytes"))
                total = fmt_bytes(gpu.get("memory_total_bytes"))
                memory_cell = f"{used}/{total}"
                temp = number(gpu.get("temperature"))
                temperature = f"{temp:.0f}°" if temp is not None else "—°"
                processes = gpu.get("processes", [])
                process_count = len(processes)
                power = number(gpu.get("power_usage_w"))
                power_limit = number(gpu.get("power_limit_w"))
                power_text = f"{power:.0f}/{power_limit:.0f}W" if power is not None and power_limit is not None else "—W"
                graphics_clock = number(gpu.get("clock_graphics_mhz"))
                memory_clock = number(gpu.get("clock_memory_mhz"))
                clock_text = (
                    f"{graphics_clock:.0f}/{memory_clock:.0f}"
                    if graphics_clock is not None and memory_clock is not None else "—/—"
                )
                if compact:
                    row: list[tuple[str, int, int]] = [
                        (f" {index:>2} ", row_color, curses.A_BOLD),
                        (model_cell, row_color, 0),
                        (f" {fmt_percent(gpu_util):>4} ", row_color, curses.A_BOLD),
                    ]
                else:
                    row = [
                        (f"  {index:>2}  ", row_color, curses.A_BOLD),
                        (model_cell, row_color, 0),
                        (f"  {fmt_percent(gpu_util):>4}  ", row_color, curses.A_BOLD),
                    ]
                row.extend(bar_segments(gpu_util, gpu_color))
                row.append((f"  {memory_cell:>11}  ", row_color, 0))
                row.append((" MEM:", row_color, curses.A_BOLD))
                row.extend(bar_segments(mem_pct, mem_color))
                row.append((f" {temperature:>4}" if compact else f"  {temperature:>4}", row_color, 0))
                if not compact:
                    fan = fmt_percent(gpu.get("fan_speed"))
                    if width >= 132:
                        row.extend([
                            (f"  {fan:>4}  {power_text:>9}", row_color, 0),
                        ])
                        if width >= 150:
                            row.append((f"  {clock_text:>9}", row_color, 0))
                    else:
                        row.append((f"  {power_text:>5}", row_color, 0))
                row.append((f" {process_count:>2}p" if compact else f"  {process_count:>2}p", row_color, 0))
                gpu_panel(*row)

                for process in processes:
                    pid = fmt_index(process.get("pid"))
                    memory = process.get("gpu_memory_human") or fmt_bytes(process.get("gpu_memory_bytes"))
                    command = str(process.get("command") or "unknown").replace("\n", " ")
                    user = str(process.get("user") or "")
                    remote_user = str(server.get("remote_user") or "")
                    owner_uid = number(process.get("uid"))
                    remote_uid = number(server.get("remote_uid"))
                    process_memory_bytes = number(process.get("gpu_memory_bytes"))
                    device_memory_bytes = number(gpu.get("memory_total_bytes"))
                    if owner_uid is not None and remote_uid is not None:
                        is_mine = int(owner_uid) == int(remote_uid)
                    else:
                        is_mine = bool(
                            user.strip()
                            and remote_user.strip()
                            and user.strip().casefold() == remote_user.strip().casefold()
                        )
                    process_rows.append({
                        "server_index": server_index,
                        "server": str(name),
                        "gpu": index,
                        "gpu_name": str(gpu.get("name") or "NVIDIA GPU"),
                        "gpu_uuid": str(gpu.get("uuid") or ""),
                        "gpu_bus_id": str(gpu.get("bus_id") or ""),
                        "gpu_utilization": number(gpu.get("gpu_utilization")),
                        "gpu_device_memory_used_bytes": number(gpu.get("memory_used_bytes")),
                        "gpu_device_memory_total_bytes": number(gpu.get("memory_total_bytes")),
                        "gpu_temperature": number(gpu.get("temperature")),
                        "gpu_power_usage_w": number(gpu.get("power_usage_w")),
                        "gpu_power_limit_w": number(gpu.get("power_limit_w")),
                        "pid": pid,
                        "gpu_memory_bytes": number(process.get("gpu_memory_bytes")),
                        "gpu_memory_percent": (
                            process_memory_bytes / device_memory_bytes * 100
                            if process_memory_bytes is not None and device_memory_bytes and device_memory_bytes > 0 else None
                        ),
                        "gpu_memory_human": memory,
                        "gpu_sm_utilization": number(process.get("gpu_sm_utilization")),
                        "gpu_memory_utilization": number(process.get("gpu_memory_utilization")),
                        "cpu_percent": number(process.get("cpu_percent")),
                        "host_memory_bytes": number(process.get("host_memory_bytes")),
                        "host_memory_human": process.get("host_memory_human"),
                        "host_memory_percent": number(process.get("host_memory_percent")),
                        "running_time": str(process.get("running_time") or ""),
                        "gpu_context_type": str(process.get("gpu_context_type") or ""),
                        "ppid": process.get("ppid"),
                        "user": user,
                        "remote_user": remote_user,
                        "mine": is_mine,
                        "command": command,
                    })
        else:
            address = server.get("address") or server.get("connection") or ""
            line(
                ("● ", colors["red"], curses.A_BOLD),
                (str(name), colors["red"], curses.A_BOLD),
                (f"  {address}  OFFLINE", colors["muted"], 0),
            )
            message = str(server.get("error") or "Unknown SSH error").replace("\n", " ")
            wrap_width = max(24, min(150, width - 6))
            for segment in textwrap.wrap(message, wrap_width) or ["Unknown SSH error"]:
                line((f"   {segment}", colors["red"], 0))
    process_rows.sort(key=lambda item: (
        item["server_index"],
        number(item["gpu"]) if number(item["gpu"]) is not None else -1,
        number(item["pid"]) if number(item["pid"]) is not None else -1,
    ))
    return lines, process_rows


def process_columns(width: int) -> tuple[int, int, int, bool]:
    full_metrics = width >= 110
    host_width = max(8, min(18, width // 8))
    user_width = max(6, min(14, width // 12))
    fixed_width = 51 if full_metrics else 30
    command_width = max(4, width - fixed_width - host_width - user_width)
    return host_width, user_width, command_width, full_metrics


def process_header(width: int) -> str:
    host_width, user_width, _, full_metrics = process_columns(width)
    metrics = f" {'%CPU':>5} {'%MEM':>5} {'TIME':>8}" if full_metrics else ""
    return f"  {'HOST':<{host_width}} {'GPU#':>3} {'PID':>7} {'VRAM':>6} {'%SM':>4}{metrics} {'USER':<{user_width}} COMMAND"


def process_line(
    process: dict[str, Any],
    width: int,
    colors: dict[str, int],
    selected: bool = False,
) -> list[tuple[str, int, int]]:
    host_width, user_width, command_width, full_metrics = process_columns(width)
    mine = bool(process.get("mine"))
    row_color = colors["white"] if mine else colors["muted"]
    user_color = colors["cyan"] if mine else colors["muted"]
    host = str(process.get("server") or "")[:host_width]
    gpu = fmt_index(process.get("gpu"))
    pid = str(process.get("pid") or "?")
    memory = fmt_bytes(process.get("gpu_memory_bytes"))
    sm = fmt_percent(process.get("gpu_sm_utilization"))
    user = str(process.get("user") or "?")[:user_width]
    command = str(process.get("command") or "unknown")
    if len(command) > command_width:
        command = command[: max(1, command_width - 1)] + "…"
    metrics = ""
    if full_metrics:
        cpu = fmt_percent(process.get("cpu_percent"))
        host_memory = fmt_percent(process.get("host_memory_percent"))
        running_time = compact_runtime(process.get("running_time"))
        metrics = f" {cpu:>5} {host_memory:>5} {running_time:>8}"
    segments = [
        ("  " + f"{host:<{host_width}}" + " ", row_color, 0),
        (f"{gpu:>3} {pid:>7} {memory:>6} {sm:>4}{metrics} ", row_color, 0),
        (f"{user:<{user_width}}", user_color, curses.A_BOLD if mine else 0),
        (" " + command, row_color, 0),
    ]
    if selected:
        segments = [(text, color, attributes | curses.A_REVERSE) for text, color, attributes in segments]
    return segments


SORT_SHORTCUTS = {
    "n": ("gpu", False, "GPU↑"), "N": ("gpu", True, "GPU↓"),
    "u": ("user", False, "USER↑"), "U": ("user", True, "USER↓"),
    "p": ("pid", True, "PID↓"), "P": ("pid", False, "PID↑"),
    "g": ("gpu_memory_bytes", True, "VRAM↓"), "G": ("gpu_memory_bytes", False, "VRAM↑"),
    "s": ("gpu_sm_utilization", True, "%SM↓"), "S": ("gpu_sm_utilization", False, "%SM↑"),
    "c": ("cpu_percent", True, "%CPU↓"), "C": ("cpu_percent", False, "%CPU↑"),
    "m": ("host_memory_percent", True, "%MEM↓"), "M": ("host_memory_percent", False, "%MEM↑"),
    "t": ("running_time_seconds", True, "TIME↓"), "T": ("running_time_seconds", False, "TIME↑"),
}
SORT_CYCLE = ("n", "u", "p", "g", "s", "c", "m", "t")


def sort_process_rows(
    processes: list[dict[str, Any]],
    field: str | None,
    descending: bool = False,
) -> list[dict[str, Any]]:
    if field is None:
        return processes
    if field == "gpu":
        server_indices = sorted({int(number(process.get("server_index")) or 0) for process in processes})
        ordered = []
        for server_index in server_indices:
            group = [process for process in processes if int(number(process.get("server_index")) or 0) == server_index]
            ordered.extend(sorted(group, key=lambda process: (
                number(process.get("gpu")) if number(process.get("gpu")) is not None else -1,
                number(process.get("pid")) if number(process.get("pid")) is not None else -1,
            ), reverse=descending))
        return ordered

    def value_for(process: dict[str, Any]) -> Any:
        if field == "user":
            value = str(process.get("user") or "").casefold()
            return value or None
        if field == "pid":
            return number(process.get("pid"))
        if field == "running_time_seconds":
            # The process collector reports elapsed time as HH:MM:SS or D-HH:MM:SS.
            raw = str(process.get("running_time") or "")
            try:
                days, clock = raw.split("-", 1) if "-" in raw else ("0", raw)
                parts = [int(part) for part in clock.split(":")]
                if len(parts) == 2:
                    hours, minutes, seconds = 0, parts[0], parts[1]
                else:
                    hours, minutes, seconds = parts
                return int(days) * 86400 + int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            except (ValueError, TypeError):
                return None
        return number(process.get(field))

    present = [process for process in processes if value_for(process) is not None]
    missing = [process for process in processes if value_for(process) is None]
    present.sort(key=value_for, reverse=descending)
    return present + missing


def filter_process_rows(processes: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    query = query.strip().casefold()
    if not query:
        return processes
    return [
        process for process in processes
        if query in " ".join(str(process.get(field) or "") for field in (
            "server", "gpu", "pid", "user", "command", "gpu_context_type",
        )).casefold()
    ]


def process_history_key(process: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(process.get("server_index", "")),
        str(process.get("gpu", "")),
        str(process.get("pid", "")),
    )


def update_process_history(
    processes: list[dict[str, Any]],
    histories: dict[tuple[str, str, str], dict[str, list[float]]],
    limit: int = 60,
) -> None:
    for process in processes:
        history = histories.setdefault(process_history_key(process), {"sm": [], "cpu": [], "mem": []})
        for key, field in (("sm", "gpu_sm_utilization"), ("cpu", "cpu_percent"), ("mem", "host_memory_percent")):
            value = number(process.get(field))
            if value is not None:
                history[key].append(value)
                del history[key][:-limit]


def sparkline(values: list[float], width: int = 56) -> str:
    if not values:
        return "—"
    values = values[-width:]
    maximum = max(100.0, max(values))
    blocks = "▁▂▃▄▅▆▇█"
    return "".join(blocks[min(len(blocks) - 1, int(max(0.0, item) / maximum * (len(blocks) - 1)))] for item in values)


def draw_process_detail(
    screen: Any,
    process: dict[str, Any],
    histories: dict[tuple[str, str, str], dict[str, list[float]]],
    colors: dict[str, int],
) -> None:
    height, width = screen.getmaxyx()
    screen.erase()
    title = f" PROCESS METRICS  {process.get('server', '')} · GPU #{process.get('gpu', '?')} · PID {process.get('pid', '?')} "
    safe_add(screen, 0, 0, title, width, colors["cyan"], curses.A_BOLD | curses.A_REVERSE)
    safe_add(screen, 1, 0, "Esc/Enter 返回 · r 刷新 · Space 暂停/继续 · h 帮助", width, colors["muted"], 0)
    temperature = number(process.get("gpu_temperature"))
    power = number(process.get("gpu_power_usage_w"))
    power_limit = number(process.get("gpu_power_limit_w"))
    power_text = f"{power:.0f}/{power_limit:.0f}W" if power is not None and power_limit is not None else (f"{power:.0f}W" if power is not None else "—")
    rows = [
        f"GPU         #{process.get('gpu', '?')}  {process.get('gpu_name') or 'NVIDIA GPU'}",
        f"GPU LOAD    {fmt_percent(process.get('gpu_utilization'))}    VRAM {fmt_bytes(process.get('gpu_device_memory_used_bytes'))}/{fmt_bytes(process.get('gpu_device_memory_total_bytes'))}",
        f"GPU THERMAL {temperature:.0f}°C" if temperature is not None else "GPU THERMAL —",
        f"GPU UUID    {process.get('gpu_uuid') or '—'}",
        f"GPU PCI     {process.get('gpu_bus_id') or '—'}",
        f"GPU POWER   {power_text}",
        f"USER        {process.get('user') or '—'}    TYPE {process.get('gpu_context_type') or '—'}    PPID {process.get('ppid') or '—'}",
        f"GPU MEMORY  {process.get('gpu_memory_human') or fmt_bytes(process.get('gpu_memory_bytes'))} ({fmt_percent(process.get('gpu_memory_percent'))})    MEMORY BW {fmt_percent(process.get('gpu_memory_utilization'))}",
        f"GPU SM      {fmt_percent(process.get('gpu_sm_utilization'))}    CPU {fmt_percent(process.get('cpu_percent'))}    HOST MEMORY {process.get('host_memory_human') or fmt_bytes(process.get('host_memory_bytes'))} ({fmt_percent(process.get('host_memory_percent'))})",
        f"RUNNING     {process.get('running_time') or '—'}",
        "COMMAND",
    ]
    y = 3
    for row in rows:
        safe_add(screen, y, 2, row, width - 2, colors["white"], 0)
        y += 1
    command = str(process.get("command") or "unknown")
    for segment in textwrap.wrap(command, max(20, width - 6)):
        if y >= height - 7:
            break
        safe_add(screen, y, 4, segment, width - 4, colors["white"], 0)
        y += 1

    history = histories.get(process_history_key(process), {})
    y = max(y + 1, height - 7)
    for label, key in (("GPU SM", "sm"), ("CPU", "cpu"), ("HOST MEM", "mem")):
        if y >= height - 1:
            break
        values = history.get(key, [])
        line = f"{label:<9} {sparkline(values, max(1, width - 22))}"
        if values:
            line += f"  {values[-1]:.1f}%"
        safe_add(screen, y, 2, line, width - 2, colors["green"] if key == "sm" else colors["cyan"], 0)
        y += 1
    screen.refresh()


def draw_help(screen: Any, colors: dict[str, int]) -> None:
    height, width = screen.getmaxyx()
    screen.erase()
    safe_add(screen, 0, 0, " GPU FLEET  HELP ", width, colors["cyan"], curses.A_BOLD | curses.A_REVERSE)
    rows = [
        "q 退出 · r 立即刷新 · Space 暂停/继续 · h 或 ? 返回",
        "默认 ↑/↓ 或 j/k 选择进程；Tab 切换到 GPU 列表浏览",
        "鼠标单击进程行可选中并高亮；双击进程行打开详情",
        "Home/End 跳到首/末进程；[ / ] 滚动进程；PgUp/PgDn 翻页",
        "f 输入筛选词（匹配服务器、GPU、PID、用户、命令）；留空清除",
        "o 后输入排序键：n/N GPU，u/U 用户，p/P PID，g/G 显存",
        "                     s/S %SM，c/C %CPU，m/M %MEM，t/T 运行时间",
        ", / . 选择相邻排序列；/ 反转当前排序方向",
        "Enter 查看所选进程详情和最近采样曲线 · Esc 返回",
        "进程详情和筛选均为只读；所有数据来自远端 nvitop 或 nvidia-smi/ps。",
    ]
    for index, row in enumerate(rows, start=2):
        if index >= height - 1:
            break
        safe_add(screen, index, 2, row, width - 2, colors["white"], 0)
    safe_add(screen, height - 1, 0, "按 h、? 或 Esc 返回监控", width, colors["muted"], 0)
    screen.refresh()


def prompt_process_filter(screen: Any, current: str, colors: dict[str, int]) -> str:
    height, width = screen.getmaxyx()
    prompt = f"Filter process (now: {current or 'all'}; empty clears): "
    try:
        screen.move(height - 1, 0)
        screen.clrtoeol()
    except curses.error:
        pass
    safe_add(screen, height - 1, 0, prompt, width, colors["yellow"], curses.A_BOLD)
    screen.refresh()
    try:
        curses.curs_set(1)
    except curses.error:
        pass
    curses.echo()
    screen.timeout(-1)
    try:
        start = min(len(prompt), max(0, width - 2))
        raw = screen.getstr(height - 1, start, max(1, width - start - 1))
    except curses.error:
        return current
    finally:
        curses.noecho()
        screen.timeout(200)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
    return raw.decode("utf-8", errors="replace").strip()


def gpu_scroll_limit(blocks: list[tuple[bool, list[list[tuple[str, int, int]]]]]) -> int:
    """Keep the scroll range on useful rows, skipping trailing host separators."""
    for index in range(len(blocks) - 1, -1, -1):
        is_gpu, block_lines = blocks[index]
        if is_gpu or any(text.strip() for segments in block_lines for text, _, _ in segments):
            return index
    return 0


def draw(
    screen: Any,
    lines: list[tuple[bool, list[list[tuple[str, int, int]]]]],
    processes: list[dict[str, Any]],
    gpu_scroll: int,
    process_scroll: int,
    status_text: str,
    summary_text: str,
    colors: dict[str, int],
    process_total: int | None = None,
    process_filter: str = "",
    sort_label: str = "GPU↑",
    process_focus: bool = True,
    selected_process: int = 0,
) -> tuple[int, int]:
    height, width = screen.getmaxyx()
    if height < 13 or width < 64:
        screen.erase()
        safe_add(screen, 0, 0, "请将终端调整到至少 64×13。", width, colors["yellow"], curses.A_BOLD)
        screen.refresh()
        return 0, 0

    # Keep enough space for a complete GPU frame and its host heading. Larger
    # fleets can still be browsed with the GPU scroll keys.
    process_capacity = min(len(processes) or 1, max(1, height - 12))
    process_top = height - process_capacity - 3
    separator_row = process_top - 1
    gpu_data = lines[1:]
    max_gpu_scroll = gpu_scroll_limit(gpu_data)
    gpu_scroll = max(0, min(gpu_scroll, max_gpu_scroll))
    max_process_scroll = max(0, len(processes) - process_capacity)
    process_scroll = max(0, min(process_scroll, max_process_scroll))
    screen.erase()
    header = f" GPU FLEET  {summary_text}"
    header += f"   │   {status_text} "
    safe_add(screen, 0, 0, header, width, colors["cyan"], curses.A_BOLD | curses.A_REVERSE)
    safe_add(screen, 1, 0, " q退出 r刷新 Space暂停 ↑↓选进程 鼠标单击选中/双击详情 Tab切换GPU f筛选 o排序 h帮助", width, colors["muted"], 0)
    safe_add(screen, 2, 0, "─" * max(1, width - 1), width, colors["muted"], 0)
    if lines:
        safe_add_segments(screen, 3, lines[0][1][0], width)
    row = 4
    gpu_number = sum(is_gpu for is_gpu, _ in gpu_data[:gpu_scroll])
    visible_gpu_numbers: list[int] = []
    gpu_table_continues = False
    for block_index in range(gpu_scroll, len(gpu_data)):
        is_gpu, block_lines = gpu_data[block_index]
        previous_is_gpu = block_index > 0 and gpu_data[block_index - 1][0]
        next_is_gpu = block_index + 1 < len(gpu_data) and gpu_data[block_index + 1][0]
        needed_rows = len(block_lines) + (1 if is_gpu else 0) + (1 if is_gpu and not next_is_gpu else 0)
        if row + needed_rows > separator_row:
            break
        if is_gpu:
            gpu_number += 1
            visible_gpu_numbers.append(gpu_number)
            left = "├" if previous_is_gpu else "┌"
            right = "┤" if previous_is_gpu else "┐"
            safe_add_segments(screen, row, [(
                left + "─" * max(1, width - 3) + right,
                colors["muted"], 0,
            )], width)
            row += 1
        for segments in block_lines:
            safe_add_segments(screen, row, segments, width)
            row += 1
        if is_gpu and not next_is_gpu:
            safe_add_segments(screen, row, [(
                "└" + "─" * max(1, width - 3) + "┘", colors["muted"], 0,
            )], width)
            row += 1
        gpu_table_continues = is_gpu and next_is_gpu
    if gpu_table_continues:
        bottom_row = min(row, separator_row)
        safe_add_segments(screen, bottom_row, [(
            "└" + "─" * max(1, width - 3) + "┘", colors["muted"], 0,
        )], width)
        if bottom_row < separator_row:
            safe_add(screen, separator_row, 0, "─" * max(1, width - 1), width, colors["muted"], 0)
    else:
        safe_add(screen, separator_row, 0, "─" * max(1, width - 1), width, colors["muted"], 0)

    total = len(processes) if process_total is None else process_total
    count = f"{len(processes)}" if len(processes) == total else f"{len(processes)}/{total}"
    if processes:
        first_process = process_scroll + 1
        last_process = min(len(processes), process_scroll + process_capacity)
        title = f" PROCESSES  {count}  [{first_process}-{last_process}] "
    else:
        title = f" PROCESSES  {count}  [0] "
    safe_add_segments(screen, process_top, [
        (title, colors["cyan"], curses.A_BOLD),
        ("  USER ", colors["muted"], 0),
        ("当前账号", colors["cyan"], curses.A_BOLD),
        (" · ", colors["muted"], 0),
        ("其他用户", colors["muted"], curses.A_DIM),
        ("  · 进程焦点" if process_focus else "", colors["yellow"], curses.A_BOLD if process_focus else 0),
        (f"  · SORT {sort_label}", colors["muted"], 0),
        (f"  · FILTER {process_filter}" if process_filter else "", colors["yellow"], 0),
    ], width)
    safe_add(screen, process_top + 1, 0, process_header(width), width, colors["muted"], curses.A_BOLD)
    if processes:
        visible_processes = processes[process_scroll : process_scroll + process_capacity]
        for row, process in enumerate(visible_processes, start=process_top + 2):
            index = process_scroll + row - (process_top + 2)
            safe_add_segments(screen, row, process_line(
                process, width, colors,
                selected=process_focus and index == selected_process,
            ), width)
    else:
        safe_add(screen, process_top + 2, 0, "  — no compute processes —", width, colors["muted"], curses.A_DIM)

    gpu_total = sum(is_gpu for is_gpu, _ in gpu_data)
    gpu_range = (
        f"{visible_gpu_numbers[0]}-{visible_gpu_numbers[-1]}"
        if visible_gpu_numbers else "0"
    )
    process_range = (
        f"{process_scroll + 1}-{min(len(processes), process_scroll + process_capacity)}"
        if processes else "0"
    )
    footer = f" GPU {gpu_range}/{gpu_total}  PROC {process_range}/{len(processes)}  {status_text} "
    safe_add(screen, height - 1, 0, footer, width, colors["muted"], 0)
    screen.refresh()
    return max_gpu_scroll, max_process_scroll


def safe_add(screen: Any, y: int, x: int, text: str, width: int, color: int, attributes: int) -> None:
    if width <= x:
        return
    clipped = text[: max(0, width - x - 1)]
    try:
        style = curses.color_pair(color) | attributes
        if color == 6:
            style |= curses.A_DIM
        screen.addnstr(y, x, clipped, len(clipped), style)
    except curses.error:
        pass


def safe_add_segments(screen: Any, y: int, segments: list[tuple[str, int, int]], width: int) -> None:
    x = 0
    for text, color, attributes in segments:
        if x >= width - 1:
            break
        clipped = text[: max(0, width - x - 1)]
        try:
            style = curses.color_pair(color) | attributes
            if color == 6:
                style |= curses.A_DIM
            screen.addnstr(y, x, clipped, len(clipped), style)
        except curses.error:
            pass
        x += len(clipped)


def collect_once(servers: list[dict[str, str]]) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=max(1, min(12, len(servers)))) as pool:
        futures = [pool.submit(query_server, server) for server in servers]
        return [future.result() for future in futures]


def print_snapshot(results: list[dict[str, Any]]) -> None:
    colors = {"cyan": 0, "green": 0, "yellow": 0, "red": 0, "white": 0, "muted": 0}
    print("GPU FLEET  " + fleet_summary(results))
    lines, processes = build_content(results, 140, colors)
    for index, (is_gpu, block_lines) in enumerate(lines):
        previous_is_gpu = index > 0 and lines[index - 1][0]
        next_is_gpu = index + 1 < len(lines) and lines[index + 1][0]
        if is_gpu:
            edge = "├" if previous_is_gpu else "┌"
            print(edge + "─" * 137 + "┤")
        for segments in block_lines:
            print("".join(segment[0] for segment in segments))
        if is_gpu and not next_is_gpu:
            print("└" + "─" * 137 + "┘")
    print("PROCESSES")
    print(process_header(140))
    for process in processes:
        print("".join(segment[0] for segment in process_line(process, 140, colors)))
    if not processes:
        print("  — no compute processes —")


def write_telemetry_snapshot(path: str, results: list[dict[str, Any]]) -> None:
    """Atomically publish the latest private live snapshot for the native app."""
    destination = Path(path).expanduser()
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(destination.parent, 0o700)
        temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
        document = {"timestamp": time.time(), "servers": results}
        temporary.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except (OSError, UnboundLocalError):
            pass


def fleet_summary(servers: list[dict[str, Any]]) -> str:
    online = sum(
        bool(server.get("ok")) and not server.get("_refresh_failures")
        for server in servers
    )
    retrying = sum(int(server.get("_refresh_failures", 0) or 0) > 0 for server in servers)
    gpus = [
        gpu
        for server in servers if server.get("ok")
        for gpu in server.get("gpus", []) if not gpu.get("error")
    ]
    process_count = sum(len(gpu.get("processes", [])) for gpu in gpus)
    utilizations = [
        number(gpu.get("gpu_utilization")) for gpu in gpus
        if number(gpu.get("gpu_utilization")) is not None
    ]
    average = sum(utilizations) / len(utilizations) if utilizations else None
    summary = f"{online}/{len(servers)} HOST  {len(gpus)} GPU  {process_count} PROC"
    if average is not None:
        summary += f"  AVG {average:.0f}%"
    if retrying:
        summary += f"  RETRY {retrying}"
    return summary


def run_tui(screen: Any, servers: list[dict[str, str]], interval: float, telemetry_file: str | None = None) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.keypad(True)
    # Poll keyboard/input more often for sub-second refresh settings.
    screen.timeout(min(200, max(25, int(interval * 500))))
    try:
        mouse_events = getattr(curses, "ALL_MOUSE_EVENTS", 0)
        if mouse_events:
            curses.mousemask(mouse_events)
        if hasattr(curses, "mouseinterval"):
            curses.mouseinterval(350)
    except curses.error:
        pass
    if curses.has_colors():
        curses.start_color()
        try:
            curses.use_default_colors()
        except curses.error:
            pass
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_RED, -1)
        curses.init_pair(5, curses.COLOR_WHITE, -1)
        curses.init_pair(6, curses.COLOR_WHITE, -1)
        colors = {"cyan": 1, "green": 2, "yellow": 3, "red": 4, "white": 5, "muted": 6}
    else:
        colors = {"cyan": 0, "green": 0, "yellow": 0, "red": 0, "white": 0, "muted": 0}

    executor = ThreadPoolExecutor(max_workers=max(1, min(12, len(servers))))
    results: list[dict[str, Any]] = [
        {
            "name": server["name"],
            "connection": server["connection"],
            "ok": False,
            "error": "Connecting…",
        }
        for server in servers
    ]
    failure_streak = [0 for _ in servers]

    def accept_refresh(index: int, latest: dict[str, Any]) -> None:
        if latest.get("ok"):
            failure_streak[index] = 0
            results[index] = latest
            return
        failure_streak[index] += 1
        previous = results[index]
        if previous.get("ok") and failure_streak[index] < 3:
            # Keep the last good snapshot through brief transport hiccups.
            # The heading marks it stale while the next refresh reconnects.
            results[index] = {
                **previous,
                "_refresh_failures": failure_streak[index],
                "_refresh_error": latest.get("error"),
            }
        else:
            results[index] = latest
    pending: list[tuple[int, Any]] = []
    auto_refresh = True
    force_refresh = True
    refresh_after_pending = False
    batch_active = False
    next_refresh = 0.0
    last_updated: float | None = None
    scroll = 0
    process_scroll = 0
    process_focus = True
    selected_process = 0
    process_filter = ""
    sort_field: str | None = None
    sort_descending = False
    sort_label = "GPU/PID"
    sort_shortcut: str | None = None
    sort_prefix = False
    detail_open = False
    help_open = False
    histories: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    history_updated_at: float | None = None

    try:
        while True:
            now = time.monotonic()
            if not pending and (force_refresh or (auto_refresh and now >= next_refresh)):
                pending = [
                    (index, executor.submit(query_server, server))
                    for index, server in enumerate(servers)
                ]
                batch_active = True
                force_refresh = False
                next_refresh = now + interval
            remaining = []
            for index, future in pending:
                if future.done():
                    try:
                        accept_refresh(index, future.result())
                    except Exception as error:
                        accept_refresh(index, {
                            "name": servers[index]["name"],
                            "connection": servers[index]["connection"],
                            "ok": False,
                            "error": str(error),
                        })
                else:
                    remaining.append((index, future))
            pending = remaining
            if not pending and batch_active:
                last_updated = time.time()
                batch_active = False
                if telemetry_file:
                    write_telemetry_snapshot(telemetry_file, results)
                if refresh_after_pending:
                    refresh_after_pending = False
                    force_refresh = True

            updated = time.strftime("%H:%M:%S", time.localtime(last_updated)) if last_updated else "connecting"
            mode = "AUTO" if auto_refresh else "PAUSED"
            status = f"{mode} · {updated} · {interval:g}s"
            lines, processes = build_content(results, screen.getmaxyx()[1], colors)
            if last_updated is not None and last_updated != history_updated_at:
                update_process_history(processes, histories)
                history_updated_at = last_updated
            process_total = len(processes)
            processes = filter_process_rows(processes, process_filter)
            processes = sort_process_rows(processes, sort_field, sort_descending)
            selected_process = min(selected_process, max(0, len(processes) - 1))
            height, width = screen.getmaxyx()
            process_capacity = min(len(processes) or 1, max(1, height - 12))
            if process_focus and processes:
                if selected_process < process_scroll:
                    process_scroll = selected_process
                elif selected_process >= process_scroll + process_capacity:
                    process_scroll = selected_process - process_capacity + 1
            if help_open:
                draw_help(screen, colors)
                max_gpu_scroll, max_process_scroll = gpu_scroll_limit(lines[1:]), max(0, len(processes) - process_capacity)
            elif detail_open and processes:
                draw_process_detail(screen, processes[selected_process], histories, colors)
                max_gpu_scroll, max_process_scroll = gpu_scroll_limit(lines[1:]), max(0, len(processes) - process_capacity)
            else:
                max_gpu_scroll, max_process_scroll = draw(
                    screen, lines, processes, scroll, process_scroll, status,
                    fleet_summary(results), colors, process_total, process_filter,
                    sort_label, process_focus, selected_process,
                )
            scroll = min(scroll, max_gpu_scroll)
            process_scroll = min(process_scroll, max_process_scroll)

            key = screen.getch()
            key_character = chr(key) if 0 <= key <= 0x10FFFF else ""
            if sort_prefix and key not in (-1, ord("o")) and key_character not in SORT_SHORTCUTS:
                sort_prefix = False
            if key == curses.KEY_MOUSE:
                try:
                    _, _, mouse_y, _, mouse_state = curses.getmouse()
                except curses.error:
                    continue
                left_click_mask = 0
                for event_name in ("BUTTON1_PRESSED", "BUTTON1_RELEASED", "BUTTON1_CLICKED", "BUTTON1_DOUBLE_CLICKED"):
                    left_click_mask |= getattr(curses, event_name, 0)
                if mouse_state & left_click_mask and not help_open and not detail_open and processes:
                    height, _ = screen.getmaxyx()
                    capacity = min(len(processes), max(1, height - 12))
                    process_top = height - capacity - 3
                    first_row = process_top + 2
                    visible_count = min(capacity, len(processes) - process_scroll)
                    if first_row <= mouse_y < first_row + visible_count:
                        selected_process = process_scroll + mouse_y - first_row
                        process_focus = True
                        if mouse_state & getattr(curses, "BUTTON1_DOUBLE_CLICKED", 0):
                            detail_open = True
                continue
            elif key in (ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                if pending:
                    refresh_after_pending = True
                else:
                    force_refresh = True
            elif key == ord(" "):
                auto_refresh = not auto_refresh
                if auto_refresh and not pending:
                    force_refresh = True
            elif help_open:
                if key in (27, ord("h"), ord("?")):
                    help_open = False
            elif detail_open:
                if key in (27, curses.KEY_ENTER, ord("\n"), ord("\r")):
                    detail_open = False
                elif key in (ord("h"), ord("?")):
                    help_open = True
            elif key in (ord("h"), ord("?")):
                help_open = True
            elif key == ord("\t"):
                process_focus = not process_focus
            elif key in (curses.KEY_UP, ord("k")):
                if process_focus and processes:
                    selected_process = max(0, selected_process - 1)
                else:
                    scroll = max(0, scroll - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                if process_focus and processes:
                    selected_process = min(len(processes) - 1, selected_process + 1)
                else:
                    scroll += 1
            elif key == curses.KEY_HOME:
                if process_focus:
                    selected_process = 0
                    process_scroll = 0
                else:
                    scroll = 0
            elif key == curses.KEY_END:
                if process_focus and processes:
                    selected_process = len(processes) - 1
                else:
                    scroll = max_gpu_scroll
            elif key == curses.KEY_PPAGE:
                page = max(1, screen.getmaxyx()[0] - 5)
                if process_focus and processes:
                    selected_process = max(0, selected_process - page)
                else:
                    scroll = max(0, scroll - page)
            elif key == curses.KEY_NPAGE:
                page = max(1, screen.getmaxyx()[0] - 5)
                if process_focus and processes:
                    selected_process = min(len(processes) - 1, selected_process + page)
                else:
                    scroll += page
            elif key == ord("["):
                process_scroll = max(0, process_scroll - 1)
                selected_process = min(selected_process, process_scroll + process_capacity - 1)
            elif key == ord("]"):
                process_scroll += 1
                selected_process = max(selected_process, process_scroll)
            elif key == ord("/"):
                if sort_shortcut is None:
                    sort_shortcut = "N"
                else:
                    sort_shortcut = sort_shortcut.swapcase()
                sort_field, sort_descending, sort_label = SORT_SHORTCUTS[sort_shortcut]
            elif key == ord("f"):
                process_filter = prompt_process_filter(screen, process_filter, colors)
                selected_process = 0
                process_scroll = 0
            elif key in (ord(","), ord(".")):
                current_index = SORT_CYCLE.index(sort_shortcut.lower()) if sort_shortcut else 0
                step = 1 if key == ord(".") else -1
                sort_index = (current_index + step) % len(SORT_CYCLE)
                candidate = SORT_CYCLE[sort_index]
                if SORT_SHORTCUTS[candidate][1] != sort_descending:
                    candidate = candidate.upper()
                sort_shortcut = candidate
                sort_field, sort_descending, sort_label = SORT_SHORTCUTS[sort_shortcut]
            elif key == ord("o"):
                sort_prefix = True
            elif sort_prefix:
                sort_prefix = False
                if key_character in SORT_SHORTCUTS:
                    sort_shortcut = key_character
                    sort_field, sort_descending, sort_label = SORT_SHORTCUTS[key_character]
                    selected_process = 0
                    process_scroll = 0
            elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")) and processes:
                if not process_focus:
                    process_focus = True
                detail_open = True
            elif key == 27:
                if process_filter:
                    process_filter = ""
                    selected_process = 0
                    process_scroll = 0
                else:
                    process_focus = False
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="nvitop-style terminal dashboard for multiple SSH GPU servers")
    parser.add_argument(
        "--server",
        action="append",
        help="Add a server, optionally as NAME=CONNECTION. May be repeated; without it, use the three built-in servers.",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds (0.25-60; default: 2)")
    parser.add_argument("--telemetry-file", help=argparse.SUPPRESS)
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit without opening the TUI")
    parser.add_argument(
        "--validate-connection",
        metavar="STRING",
        help="Validate a connection locally, print a JSON result, and exit without connecting.",
    )
    args = parser.parse_args()
    if args.validate_connection is not None:
        try:
            user, host, port = parse_connection(args.validate_connection)
        except ValueError as error:
            print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
            raise SystemExit(2)
        print(json.dumps({"ok": True, "user": user, "host": host, "port": port}, ensure_ascii=False))
        return
    if args.interval < 0.25 or args.interval > 60:
        parser.error("--interval must be between 0.25 and 60 seconds")
    try:
        servers = parse_servers(args.server)
    except ValueError as error:
        parser.error(str(error))
    if args.once:
        print_snapshot(collect_once(servers))
        return
    try:
        curses.wrapper(run_tui, servers, args.interval, args.telemetry_file)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
