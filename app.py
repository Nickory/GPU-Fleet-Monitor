#!/usr/bin/env python3
"""Local, read-only SSH dashboard for nvitop GPU status."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import shlex
import subprocess
import time
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
                    })
            processes.sort(key=lambda item: (item["gpu_memory_bytes"] or 0), reverse=True)
            gpus.append({
                "index": clean(value(device, "index")),
                "name": clean(value(device, "name")),
                "uuid": clean(value(device, "uuid")),
                "bus_id": clean(value(device, "bus_id")),
                "gpu_utilization": number(device, "gpu_utilization"),
                "memory_utilization": number(device, "memory_utilization"),
                "memory_total_bytes": number(device, "memory_total"),
                "memory_used_bytes": number(device, "memory_used"),
                "temperature": number(device, "temperature"),
                "fan_speed": number(device, "fan_speed"),
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
        "--query-gpu=driver_version,index,name,uuid,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,fan.speed,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ])
    gpu_rows = list(csv.reader(io.StringIO(gpu_text)))
    gpus = []
    by_uuid = {}
    driver_version = None
    for row in gpu_rows:
        if len(row) < 12:
            continue
        driver_version = driver_version or row[0].strip()
        uuid = row[3].strip()
        total_mib = numeric_field(row, 4)
        used_mib = numeric_field(row, 5)
        gpus.append({
            "index": numeric_field(row, 1),
            "name": row[2].strip(),
            "uuid": uuid,
            "bus_id": None,
            "gpu_utilization": numeric_field(row, 6),
            "memory_utilization": numeric_field(row, 7),
            "memory_total_bytes": total_mib * 1024 * 1024 if total_mib is not None else None,
            "memory_used_bytes": used_mib * 1024 * 1024 if used_mib is not None else None,
            "temperature": numeric_field(row, 8),
            "fan_speed": numeric_field(row, 9),
            "power_usage_w": numeric_field(row, 10),
            "power_limit_w": numeric_field(row, 11),
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
        gpu["processes"].append({
            "pid": numeric_field(row, 1),
            "user": None,
            "command": row[2].strip(),
            "gpu_memory_bytes": memory_mib * 1024 * 1024 if memory_mib is not None else None,
            "gpu_memory_human": str(round(memory_mib)) + " MiB" if memory_mib is not None else None,
        })
    return {"source": "nvidia-smi", "driver_version": driver_version, "gpus": gpus}

try:
    result = collect_nvitop()
except Exception as nvitop_error:
    try:
        result = collect_nvidia_smi()
    except Exception as nvidia_error:
        raise RuntimeError("nvitop: " + str(nvitop_error) + "; nvidia-smi: " + str(nvidia_error))

print("GPU_DASHBOARD_JSON=" + json.dumps(result, ensure_ascii=False))
'''

PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>GPU 状态总览</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101512;
      --panel: #171e19;
      --panel-2: #1c251f;
      --line: #2a352d;
      --text: #edf2ed;
      --muted: #9aa89d;
      --green: #95e6ad;
      --green-2: #43bc78;
      --amber: #f1c77a;
      --red: #f18e86;
      --shadow: 0 16px 42px rgba(0,0,0,.2);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(ellipse at 8% -12%, rgba(71,139,92,.16), transparent 40rem),
        radial-gradient(ellipse at 100% 0%, rgba(70,110,80,.08), transparent 34rem),
        var(--bg);
    }
    button, input { font: inherit; }
    button { cursor: pointer; }
    .shell { width: min(1420px, calc(100% - 48px)); margin: 0 auto; padding: 38px 0 64px; }
    .topline { display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; }
    .eyebrow { display: flex; align-items: center; gap: 9px; color: var(--green); font-size: 11px; letter-spacing: .16em; font-weight: 750; text-transform: uppercase; }
    .pulse { width: 7px; height: 7px; border-radius: 50%; background: var(--green-2); box-shadow: 0 0 12px rgba(67,188,120,.8); }
    h1 { margin: 11px 0 8px; font-size: clamp(28px, 4vw, 40px); letter-spacing: -.04em; line-height: 1.1; }
    .subtitle { margin: 0; color: var(--muted); font-size: 14px; }
    .actions { display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-end; gap: 9px; padding-top: 6px; }
    .button { border: 1px solid var(--line); background: #1a231d; color: var(--text); border-radius: 10px; padding: 10px 14px; font-size: 13px; font-weight: 650; transition: .15s ease; }
    .button:hover { border-color: #506b57; background: #222e25; transform: translateY(-1px); }
    .button.primary { color: #102017; border-color: var(--green); background: var(--green); }
    .button.primary:hover { background: #b1f2c1; }
    .button.subtle { background: transparent; }
    .button:disabled { opacity: .55; cursor: wait; transform: none; }
    .stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 30px 0 22px; }
    .stat { min-height: 106px; border: 1px solid var(--line); background: linear-gradient(145deg, rgba(30,40,33,.95), rgba(21,28,23,.96)); border-radius: 14px; padding: 17px 19px; box-shadow: var(--shadow); }
    .stat-label { color: var(--muted); font-size: 12px; font-weight: 600; }
    .stat-value { margin-top: 9px; font-size: 29px; line-height: 1; letter-spacing: -.04em; font-weight: 720; }
    .stat-value small { color: var(--muted); font-size: 13px; font-weight: 550; letter-spacing: 0; margin-left: 4px; }
    .editor { border: 1px solid var(--line); background: rgba(23,30,25,.9); border-radius: 14px; margin-bottom: 28px; overflow: hidden; }
    .editor summary { display: flex; align-items: center; justify-content: space-between; gap: 12px; cursor: pointer; padding: 16px 19px; list-style: none; }
    .editor summary::-webkit-details-marker { display: none; }
    .editor-title { font-size: 13px; font-weight: 700; }
    .editor-hint { color: var(--muted); font-size: 12px; }
    .editor-body { border-top: 1px solid var(--line); padding: 17px 18px 18px; }
    .edit-row { display: grid; grid-template-columns: minmax(130px,.65fr) minmax(280px,1.8fr) auto; gap: 9px; margin-bottom: 9px; }
    .input { width: 100%; height: 41px; color: var(--text); background: #111713; border: 1px solid #303b33; border-radius: 8px; outline: none; padding: 0 11px; font-size: 12px; }
    .input:focus { border-color: #6eb883; box-shadow: 0 0 0 3px rgba(110,184,131,.12); }
    .remove { width: 41px; height: 41px; border: 1px solid var(--line); border-radius: 8px; color: #b4c0b7; background: transparent; font-size: 17px; }
    .remove:hover { color: var(--red); border-color: rgba(241,142,134,.5); }
    .editor-foot { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-top: 14px; }
    .format-note { color: var(--muted); font-size: 11px; line-height: 1.7; }
    .editor-buttons { display: flex; gap: 8px; }
    .section-head { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 13px; }
    .section-head h2 { margin: 0; font-size: 16px; letter-spacing: -.015em; }
    .updated { color: var(--muted); font-size: 12px; }
    .server-list { display: grid; gap: 15px; }
    .server { overflow: hidden; border: 1px solid var(--line); border-radius: 15px; background: linear-gradient(150deg, rgba(26,34,29,.97), rgba(20,27,22,.97)); box-shadow: var(--shadow); }
    .server-head { display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 16px 19px; border-bottom: 1px solid var(--line); }
    .server-name { font-size: 14px; font-weight: 720; }
    .server-address { margin-top: 4px; color: var(--muted); font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .server-meta { display: flex; align-items: center; gap: 13px; }
    .driver { color: var(--muted); font-size: 11px; }
    .badge { display: inline-flex; align-items: center; gap: 6px; border-radius: 99px; padding: 5px 9px; font-size: 10px; font-weight: 700; white-space: nowrap; }
    .badge:before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
    .badge.up { color: var(--green); background: rgba(80,173,108,.12); }
    .badge.down { color: var(--red); background: rgba(216,94,84,.12); }
    .gpu-grid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px; padding: 13px; }
    .gpu { min-width: 0; border: 1px solid #2b372e; border-radius: 12px; background: rgba(18,24,20,.7); padding: 15px; }
    .gpu-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; }
    .gpu-index { color: var(--green); font-size: 11px; font-weight: 750; letter-spacing: .04em; }
    .gpu-model { margin-top: 5px; font-size: 13px; font-weight: 700; line-height: 1.35; }
    .gpu-uuid { max-width: 150px; color: #78877c; font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: right; }
    .load-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 18px 0 16px; }
    .load-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 7px; }
    .load-label { color: var(--muted); font-size: 11px; }
    .load-value { font-size: 18px; font-variant-numeric: tabular-nums; font-weight: 720; }
    .track { height: 6px; overflow: hidden; background: #2b352e; border-radius: 99px; }
    .fill { height: 100%; min-width: 0; border-radius: inherit; background: linear-gradient(90deg, #59c67e, #a5edb4); transition: width .35s ease; }
    .fill.warn { background: linear-gradient(90deg, #daa84f, #f1cd82); }
    .fill.hot { background: linear-gradient(90deg, #dc776c, #f2a39a); }
    .details { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; padding: 11px 0 12px; border-top: 1px solid #263129; border-bottom: 1px solid #263129; }
    .detail-label { color: #8d9c91; font-size: 10px; }
    .detail-value { margin-top: 4px; font-size: 12px; font-weight: 650; font-variant-numeric: tabular-nums; }
    .process-head { display: flex; align-items: center; justify-content: space-between; margin: 13px 0 8px; color: #aeb9b1; font-size: 10px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }
    .process-count { color: var(--muted); font-size: 10px; font-weight: 500; letter-spacing: 0; text-transform: none; }
    .process { display: grid; grid-template-columns: minmax(45px,.35fr) minmax(0,1.6fr) minmax(64px,.5fr) minmax(65px,.5fr); gap: 8px; align-items: center; padding: 8px 0; border-top: 1px solid rgba(42,53,45,.65); }
    .process:first-of-type { border-top: 0; }
    .pid { color: var(--green); font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .command { overflow: hidden; color: #dbe3dc; font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; text-overflow: ellipsis; white-space: nowrap; }
    .proc-user,.proc-mem { overflow: hidden; color: var(--muted); font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
    .empty-process { padding: 11px 0 1px; color: #849287; font-size: 11px; }
    .server-error { margin: 14px; padding: 12px 13px; border: 1px solid rgba(241,142,134,.24); border-radius: 9px; background: rgba(90,42,39,.16); color: #e9aaa4; font-size: 12px; line-height: 1.55; overflow-wrap: anywhere; }
    .server-empty { padding: 22px 15px; color: var(--muted); font-size: 12px; }
    .notice { margin-top: 16px; color: #718076; font-size: 10px; line-height: 1.65; }
    .footer { display: flex; justify-content: space-between; gap: 16px; margin-top: 20px; padding-top: 14px; border-top: 1px solid rgba(42,53,45,.7); color: #748177; font-size: 10px; }
    .spin { display: inline-block; animation: spin .8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    @media (max-width: 900px) {
      .shell { width: min(100% - 32px, 700px); padding-top: 27px; }
      .topline { flex-direction: column; }
      .actions { justify-content: flex-start; padding-top: 0; }
      .gpu-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 620px) {
      .shell { width: calc(100% - 24px); }
      .stats { grid-template-columns: repeat(2,1fr); gap: 8px; margin-top: 21px; }
      .stat { min-height: 91px; padding: 14px; }
      .stat-value { font-size: 25px; }
      .edit-row { grid-template-columns: 1fr auto; }
      .edit-row .connection-input { grid-column: 1 / -1; grid-row: 2; }
      .edit-row .remove { grid-column: 2; grid-row: 1; }
      .server-head { align-items: flex-start; padding: 14px; }
      .server-meta { align-items: flex-end; flex-direction: column-reverse; gap: 6px; }
      .driver { text-align: right; }
      .gpu-grid { padding: 9px; }
      .gpu { padding: 13px; }
      .editor-foot { align-items: flex-start; flex-direction: column; }
      .editor-buttons { width: 100%; }
      .editor-buttons .button { flex: 1; }
      .footer { flex-direction: column; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topline">
      <div>
        <div class="eyebrow"><span class="pulse"></span> GPU FLEET MONITOR</div>
        <h1>显卡状态总览</h1>
        <p class="subtitle">通过 SSH 读取各服务器的 nvitop GPU 与进程状态</p>
      </div>
      <div class="actions">
        <button class="button subtle" id="autoButton" type="button">自动刷新：开</button>
        <button class="button primary" id="refreshButton" type="button">↻　立即刷新</button>
      </div>
    </header>

    <section class="stats" aria-label="汇总">
      <div class="stat"><div class="stat-label">在线服务器</div><div class="stat-value" id="onlineStat">—<small>台</small></div></div>
      <div class="stat"><div class="stat-label">GPU 总数</div><div class="stat-value" id="gpuStat">—<small>张</small></div></div>
      <div class="stat"><div class="stat-label">GPU 进程</div><div class="stat-value" id="processStat">—<small>个</small></div></div>
      <div class="stat"><div class="stat-label">平均 GPU 利用率</div><div class="stat-value" id="averageStat">—<small>%</small></div></div>
    </section>

    <details class="editor" id="editor" open>
      <summary><span class="editor-title">服务器列表</span><span class="editor-hint">连接信息仅保存在当前浏览器</span></summary>
      <div class="editor-body">
        <div id="editRows"></div>
        <div class="editor-foot">
          <div class="format-note">支持 ssh -p 端口 用户@主机，也支持 主机:端口。使用本机 SSH 密钥或 ssh-agent；不会保存密码。</div>
          <div class="editor-buttons">
            <button class="button subtle" id="addButton" type="button">＋ 添加服务器</button>
            <button class="button primary" id="applyButton" type="button">保存并刷新</button>
          </div>
        </div>
      </div>
    </details>

    <section>
      <div class="section-head">
        <h2>实时状态</h2>
        <div class="updated" id="updatedLabel">等待采集</div>
      </div>
      <div class="server-list" id="serverList"></div>
      <p class="notice">优先通过 nvitop Python API 采集；未安装 nvitop 时自动回退到 nvidia-smi，并在服务器卡片标明来源。面板只读取状态，不会启动、停止或修改远端进程。</p>
    </section>
    <footer class="footer"><span>本机运行 · 仅监听 127.0.0.1</span><span>数据不会上传或写入服务器</span></footer>
  </main>
  <script>
    const DEFAULTS = [
      {name: "GPU Server 01", connection: "gpu01.example.invalid:22"},
      {name: "GPU Server 02", connection: "gpu02.example.invalid:22"},
      {name: "GPU Server 03", connection: "gpu03.example.invalid:22"}
    ];
    const STORAGE_KEY = "gpu-fleet-servers-v1";
    let servers = loadServers();
    let refreshing = false;
    let autoRefresh = true;
    let lastResults = [];

    function esc(value) {
      return String(value == null ? "" : value).replace(/[&<>"']/g, function(ch) {
        return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[ch];
      });
    }
    function loadServers() {
      try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
        if (Array.isArray(saved)) return saved;
      } catch (_) {}
      return DEFAULTS.map(function(item) { return Object.assign({}, item); });
    }
    function saveServers(value) {
      servers = value;
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(servers)); } catch (_) {}
    }
    function editorRow(item) {
      return '<div class="edit-row">' +
        '<input class="input name-input" aria-label="服务器名称" placeholder="服务器名称" value="' + esc(item.name) + '">' +
        '<input class="input connection-input" aria-label="SSH 连接" placeholder="ssh -p 2222 user@host 或 host:port" value="' + esc(item.connection) + '">' +
        '<button class="remove" type="button" aria-label="移除服务器" title="移除">×</button>' +
        '</div>';
    }
    function renderEditor(items) {
      document.getElementById("editRows").innerHTML = items.map(editorRow).join("");
      document.querySelectorAll(".remove").forEach(function(button) {
        button.addEventListener("click", function() { button.closest(".edit-row").remove(); });
      });
    }
    function readEditor() {
      return Array.from(document.querySelectorAll(".edit-row")).map(function(row) {
        return {
          name: row.querySelector(".name-input").value.trim(),
          connection: row.querySelector(".connection-input").value.trim()
        };
      }).filter(function(item) { return item.connection; });
    }
    function setRefreshing(value) {
      refreshing = value;
      const button = document.getElementById("refreshButton");
      button.disabled = value;
      button.innerHTML = value ? '<span class="spin">↻</span>　采集中…' : '↻　立即刷新';
    }
    function n(value) {
      if (value === null || value === undefined || value === "") return null;
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }
    function pct(value) {
      const number = n(value);
      return number === null ? "—" : Math.round(number) + "%";
    }
    function bytes(value) {
      const number = n(value);
      if (number === null) return "—";
      const gib = number / (1024 * 1024 * 1024);
      return gib >= 1 ? gib.toFixed(gib < 10 ? 1 : 0) + " GiB" : (number / (1024 * 1024)).toFixed(0) + " MiB";
    }
    function safeMemory(gpu) {
      const used = n(gpu.memory_used_bytes);
      const total = n(gpu.memory_total_bytes);
      if (used === null || total === null || total <= 0) return "—";
      return bytes(used) + " / " + bytes(total);
    }
    function bar(value) {
      const number = n(value);
      const width = number === null ? 0 : Math.max(0, Math.min(100, number));
      const cls = width >= 90 ? "fill hot" : width >= 70 ? "fill warn" : "fill";
      return '<div class="track"><div class="' + cls + '" style="width:' + width + '%"></div></div>';
    }
    function detail(label, value) {
      return '<div><div class="detail-label">' + esc(label) + '</div><div class="detail-value">' + esc(value) + '</div></div>';
    }
    function gpuCard(gpu) {
      if (gpu.error) return '<article class="gpu"><div class="gpu-model">GPU ' + esc(gpu.index) + '</div><div class="server-error">' + esc(gpu.error) + '</div></article>';
      const index = gpu.index == null ? "?" : gpu.index;
      const model = gpu.name || "NVIDIA GPU";
      const used = n(gpu.memory_used_bytes);
      const total = n(gpu.memory_total_bytes);
      const memoryPct = used !== null && total ? (used / total * 100) : n(gpu.memory_utilization);
      const processes = Array.isArray(gpu.processes) ? gpu.processes : [];
      const rows = processes.length ? processes.map(function(process) {
        const command = process.command || "未知进程";
        return '<div class="process">' +
          '<div class="pid">' + esc(process.pid == null ? "—" : process.pid) + '</div>' +
          '<div class="command" title="' + esc(command) + '">' + esc(command) + '</div>' +
          '<div class="proc-user">' + esc(process.user || "—") + '</div>' +
          '<div class="proc-mem">' + esc(process.gpu_memory_human || bytes(process.gpu_memory_bytes)) + '</div>' +
          '</div>';
      }).join("") : '<div class="empty-process">暂无 GPU 进程</div>';
      return '<article class="gpu">' +
        '<div class="gpu-top"><div><div class="gpu-index">GPU ' + esc(index) + '</div><div class="gpu-model">' + esc(model) + '</div></div>' +
        '<div class="gpu-uuid" title="' + esc(gpu.uuid || gpu.bus_id || "") + '">' + esc(gpu.uuid || gpu.bus_id || "") + '</div></div>' +
        '<div class="load-grid">' +
          '<div><div class="load-head"><span class="load-label">GPU 利用率</span><span class="load-value">' + esc(pct(gpu.gpu_utilization)) + '</span></div>' + bar(gpu.gpu_utilization) + '</div>' +
          '<div><div class="load-head"><span class="load-label">显存利用</span><span class="load-value">' + esc(pct(memoryPct)) + '</span></div>' + bar(memoryPct) + '</div>' +
        '</div>' +
        '<div class="details">' +
          detail("显存 已用 / 总量", safeMemory(gpu)) +
          detail("温度", n(gpu.temperature) === null ? "—" : Math.round(n(gpu.temperature)) + "°C") +
          detail("风扇", pct(gpu.fan_speed)) +
          detail("功耗", n(gpu.power_usage_w) === null ? "—" : n(gpu.power_usage_w).toFixed(1) + " W") +
        '</div>' +
        '<div class="process-head"><span>GPU 进程</span><span class="process-count">' + processes.length + ' 个</span></div>' +
        (processes.length ? '<div class="process">' +
          '<div class="pid">PID</div><div class="command">命令</div><div class="proc-user">用户</div><div class="proc-mem">显存</div></div>' : '') +
        rows +
      '</article>';
    }
    function serverCard(server) {
      const badge = server.ok ? '<span class="badge up">在线</span>' : '<span class="badge down">连接失败</span>';
      const source = server.source === "nvidia-smi" ? "nvidia-smi" : "nvitop API";
      const driver = server.ok ? source + (server.driver_version ? ' · Driver ' + esc(server.driver_version) : '') : '';
      let body = '';
      if (!server.ok) {
        body = '<div class="server-error">' + esc(server.error || "连接失败") + '</div>';
      } else if (!server.gpus || !server.gpus.length) {
        body = '<div class="server-empty">已连接，但没有检测到 NVIDIA GPU。</div>';
      } else {
        body = '<div class="gpu-grid">' + server.gpus.map(gpuCard).join("") + '</div>';
      }
      return '<article class="server">' +
        '<div class="server-head"><div><div class="server-name">' + esc(server.name || "未命名服务器") + '</div><div class="server-address">' + esc(server.address || server.connection || "") + '</div></div>' +
        '<div class="server-meta">' + (driver ? '<span class="driver">' + driver + '</span>' : '') + badge + '</div></div>' +
        body +
      '</article>';
    }
    function render(data) {
      lastResults = data.servers || [];
      const allGpus = [];
      lastResults.forEach(function(server) { if (server.ok && Array.isArray(server.gpus)) allGpus.push.apply(allGpus, server.gpus); });
      const processCount = allGpus.reduce(function(total, gpu) { return total + ((gpu.processes || []).length); }, 0);
      const utilizations = allGpus.map(function(gpu) { return n(gpu.gpu_utilization); }).filter(function(value) { return value !== null; });
      const average = utilizations.length ? Math.round(utilizations.reduce(function(a,b) { return a+b; }, 0) / utilizations.length) : null;
      document.getElementById("onlineStat").innerHTML = lastResults.filter(function(s) { return s.ok; }).length + '<small>台</small>';
      document.getElementById("gpuStat").innerHTML = allGpus.length + '<small>张</small>';
      document.getElementById("processStat").innerHTML = processCount + '<small>个</small>';
      document.getElementById("averageStat").innerHTML = (average === null ? "—" : average) + '<small>%</small>';
      document.getElementById("serverList").innerHTML = lastResults.length ?
        lastResults.map(serverCard).join("") :
        '<article class="server"><div class="server-empty">还没有服务器，请在上方添加连接信息。</div></article>';
      const date = data.updated_at ? new Date(data.updated_at * 1000) : new Date();
      document.getElementById("updatedLabel").textContent = "上次更新 " + date.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"});
    }
    async function refreshNow() {
      if (refreshing) return;
      const current = readEditor();
      saveServers(current);
      if (!current.length) {
        render({servers:[], updated_at:Date.now()/1000});
        return;
      }
      setRefreshing(true);
      document.getElementById("updatedLabel").textContent = "正在连接 " + current.length + " 台服务器…";
      try {
        const response = await fetch("/api/refresh", {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({servers:current})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "请求失败");
        render(data);
      } catch (error) {
        document.getElementById("updatedLabel").textContent = "刷新失败：" + error.message;
      } finally {
        setRefreshing(false);
      }
    }
    document.getElementById("refreshButton").addEventListener("click", refreshNow);
    document.getElementById("applyButton").addEventListener("click", refreshNow);
    document.getElementById("addButton").addEventListener("click", function() {
      const draft = readEditor();
      draft.push({name:"新服务器", connection:""});
      renderEditor(draft);
      const inputs = document.querySelectorAll(".connection-input");
      if (inputs.length) inputs[inputs.length - 1].focus();
    });
    document.getElementById("autoButton").addEventListener("click", function() {
      autoRefresh = !autoRefresh;
      this.textContent = "自动刷新：" + (autoRefresh ? "开" : "关");
      this.classList.toggle("primary", !autoRefresh);
    });
    renderEditor(servers);
    refreshNow();
    window.setInterval(function() { if (autoRefresh && !refreshing) refreshNow(); }, 10000);
  </script>
</body>
</html>
"""


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


def query_server(item: dict[str, Any]) -> dict[str, Any]:
    name = str(item.get("name", "")).strip()[:80] or "未命名服务器"
    raw = str(item.get("connection", "")).strip()
    started = time.monotonic()
    try:
        user, host, port = parse_connection(raw)
        target = (user + "@" if user else "") + host
        address = target + (":" + str(port) if port else "")
        command = [
            "ssh",
            "-T",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=8",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=1",
            target,
        ]
        if port is not None:
            command[1:1] = ["-p", str(port)]
        remote_command = "python3 -c " + shlex.quote(REMOTE_SCRIPT)
        command.append(remote_command)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=17,
            check=False,
        )
        marker = "GPU_DASHBOARD_JSON="
        payload = None
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(marker):
                payload = json.loads(line[len(marker):])
                break
        elapsed = round((time.monotonic() - started) * 1000)
        if payload is None:
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
            "gpus": payload.get("gpus", []),
            "elapsed_ms": elapsed,
        }
    except subprocess.TimeoutExpired:
        error = "连接或 nvitop 查询超时（17 秒）。"
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


class Handler(BaseHTTPRequestHandler):
    server_version = "GPUFleetDashboard/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/health":
            self.send_json(200, {"ok": True})
            return
        self.send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/refresh":
            self.send_json(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 32_000:
                raise ValueError("请求内容过大。")
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            servers = payload.get("servers", [])
            if not isinstance(servers, list) or len(servers) > 16:
                raise ValueError("服务器列表无效（最多支持 16 台）。")
            normalized = []
            for item in servers:
                if not isinstance(item, dict):
                    continue
                normalized.append({
                    "name": str(item.get("name", ""))[:80],
                    "connection": str(item.get("connection", ""))[:512],
                })
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error) or "请求格式无效。"})
            return

        if normalized:
            with ThreadPoolExecutor(max_workers=min(12, len(normalized))) as pool:
                results = list(pool.map(query_server, normalized))
        else:
            results = []
        self.send_json(200, {"updated_at": time.time(), "servers": results})


def main() -> None:
    parser = argparse.ArgumentParser(description="本机 GPU 状态总览")
    parser.add_argument("--port", type=int, default=8765, help="网页端口（默认 8765）")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在 1 到 65535 之间")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("GPU 状态总览已启动：http://127.0.0.1:" + str(args.port))
    print("按 Ctrl-C 停止。连接配置仅保存在浏览器中。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
