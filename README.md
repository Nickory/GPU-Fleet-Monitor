# GPU Fleet Monitor

> **EN:** One server? nvitop is great. GPUs scattered across nodes? How many SSH windows are you going to patrol?
>
> **中文：** 单台服务器用 nvitop 就够了。GPU 分散到不同节点？你打算开几个 SSH 窗口来回巡逻？

![A developer juggling several isolated GPU servers before bringing them together in one fleet dashboard](docs/images/multi-server-motivation.jpg)

## Why Multi-Node Monitoring? / 为什么需要多节点监控？

**The problem.** nvitop is excellent for the machine you are logged into. Training jobs, however, often spread across several servers: a few GPUs may be idle on host A, host B may be saturated, and host C may be running another set of processes. Checking each one means repeated SSH sessions, window switching, and mentally matching GPU numbers. At some point, the busiest part of the cluster is the person watching it.

**问题。** nvitop 很适合查看当前登录的服务器。但训练任务往往分散在多台机器上：A 机器有几张空闲卡，B 机器已经满载，C 机器又跑着另一批进程。逐台查看就得反复 SSH、切窗口、对照 GPU 编号——忙得最厉害的可能不是 GPU，而是盯屏幕的人。

**The idea.** Configure the hosts on your Mac or in the terminal, collect them in parallel, and view every GPU and process in one compact dashboard. Jobs stay on their original servers. The machines do not need to move to one node, and no central service is required.

**解决思路。** 在 Mac 图形界面或终端里配置多台主机，并行采集后在一个紧凑界面里查看所有 GPU 和进程。任务继续留在原服务器上运行，不必迁到同一个节点，也不需要额外部署中心服务。

## Features

- **One fleet view:** GPUs are grouped by server, with one row per GPU and a unified process table at the bottom.
- **Familiar nvitop-style details:** utilization, VRAM, temperature, power, and processes; your processes are highlighted while other users' rows are dimmed.
- **Minimal server setup:** parallel, read-only collection over system SSH. The monitor uses the remote nvitop Python API when available and falls back to structured nvidia-smi queries when it is not.
- **A native macOS GUI:** edit and filter hosts, choose the refresh rate, inspect connection diagnostics, set idle alerts, and view recent GPU history.

The terminal interface uses Python curses and follows the compact layout of nvitop -m full: GPUs above, the shared process table below, and colored utilization bars. It refreshes every 2 seconds by default and supports intervals as short as 0.25 seconds. Each GPU row is colored by the higher load level of GPU utilization (10% / 75%) and VRAM utilization (10% / 80%); the GPU and MEM bars use their own metrics.

## Terminal Quick Start

Requirements: Python 3.9 or newer and OpenSSH (included with macOS and most Linux distributions). No third-party Python packages are needed.

From the repository root, run:

    python3 gpu_watch.py

Print a single snapshot without opening the interactive interface:

    python3 gpu_watch.py --once

Pass one or more custom servers. When --server is provided, it replaces the bundled server list:

    python3 gpu_watch.py --server "A100=ssh -p 2222 user@example.com" --server "RTX=gpu01.example.invalid:22"

## macOS GUI

Build the app locally:

    zsh mac-app/build_app.sh

The app targets Apple Silicon and macOS 13 or newer. It uses a compact host list with native-style controls and supports light and dark appearance.

- Click a server row or its pencil button to edit it. The app validates SSH connection syntax when saving; canceling leaves the configuration unchanged.
- Search by name or address and choose which servers to monitor. Each server menu supports launching one host, copying a connection, duplicating, reordering, and removing with undo.
- Choose a refresh interval of 0.25, 0.5, 1, 2, or 5 seconds, or set a custom value from 0.25 to 60 seconds.
- The More menu supports importing and exporting JSON configuration, copying the monitor command, selecting servers in bulk, and restoring bundled examples. Imports merge hosts, skip duplicate connections, and preserve the current refresh interval.
- Click **Start Monitoring** (⌘↩) to open a new Terminal window and run the bundled monitor. Press ⌘N to add a server.
- The trends button opens connection diagnostics, idle alerts, and history. Diagnostics show collection status, response time, reconnect state, and error details; click a server row to inspect its issue.
- Idle alerts support GPU utilization thresholds of 5%, 10%, or 20%, sustained for 30 seconds, 1 minute, or 5 minutes. An alert is raised only when there are no running processes and utilization stays below the selected threshold. macOS notifications must be allowed the first time.
- The 30-minute charts show GPU utilization, VRAM utilization, and temperature, sampled every 5 seconds. The terminal monitor writes telemetry to a local private file; the GUI reads it for diagnostics, charts, and alerts without starting another SSH polling loop. Start the terminal monitor before opening these panels.

Settings are stored in ~/Library/Application Support/GPU Fleet Monitor/. Server configuration, live telemetry, and history stay on the Mac. The public source includes only reserved .example.invalid placeholder hosts. Connection syntax is validated locally; live connection status comes from the terminal monitor. The app looks for Python 3 in Homebrew, PATH, or the system and uses the same interpreter for validation and launch.

The app is ad-hoc signed. On first launch, macOS may require Control-clicking the app and choosing **Open**.

## Keyboard Controls

- **q:** quit
- **r:** refresh immediately
- **Space:** pause or resume automatic refresh
- **↑ / ↓ or j / k:** select processes by default; the selected row is highlighted. Press **Tab** to switch focus to the GPU list.
- Click a process row to select it; double-click to open details.
- **Home / End:** jump to the first or last item in the focused list.
- **[ / ]:** scroll the process table. It uses as much terminal height as possible and shows all processes when they fit.
- **f:** filter by server, GPU, PID, user, or command. Submit an empty filter to clear it.
- **o** followed by a key: sort by GPU (**n/N**), user (**u/U**), PID (**p/P**), GPU memory (**g/G**), %SM (**s/S**), %CPU (**c/C**), %MEM (**m/M**), or runtime (**t/T**).
- **, / .:** select the previous or next sort column; **/** reverses the sort direction.
- **Enter:** inspect the selected process, including its server, GPU ID/model/UUID/PCI address, GPU status, and recent samples.
- **h** or **?:** show help; **Esc:** go back or clear the filter.
- **PageUp / PageDown:** page through the focused list.

## SSH Connections

The terminal script includes three reserved placeholder hosts. Replace them before monitoring real servers. Supported formats include:

    ssh -p 2222 user@example.com
    gpu01.example.invalid:22
    user@host:port

The host:port format uses the default username and keys from your local SSH configuration. The script uses SSH BatchMode: it will not prompt for or store passwords. Make sure your SSH key or agent can access the servers, and accept each host fingerprint in a regular terminal first.

If remote Python can import nvitop, the monitor uses its Python API for GPU and process data. Otherwise, it falls back to structured nvidia-smi queries and marks the source in the server header. In fallback mode, nvidia-smi pmon supplies per-process GPU utilization and ps supplies process owner, CPU, host memory, and runtime. Unsupported driver or permission fields are shown as dashes.

SSH timeouts appear in the corresponding server group. For unknown host fingerprints or authentication failures, fix SSH access in a regular terminal first.

## Data and Privacy

The dashboard shows GPU model, utilization, VRAM, temperature, fan, power, and driver version. Its unified process table includes server, GPU ID, PID, command, user, GPU memory, GPU SM, CPU, host-memory percentage, and runtime. Process details include GPU model, UUID, PCI address, utilization, memory, temperature, power, and the latest 60 samples.

The current SSH account's processes are matched by UID and highlighted; other accounts are dimmed. The monitor is read-only: it does not start, stop, or modify server processes.

An optional web interface is also available:

    python3 app.py
