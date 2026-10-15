import AppKit
import Charts
import Combine
import SwiftUI
import UniformTypeIdentifiers
import UserNotifications

struct ServerConfig: Codable, Identifiable, Equatable {
    var id: UUID = UUID()
    var name: String
    var connection: String
    var enabled: Bool = true

    var cleaned: ServerConfig {
        var result = self
        result.name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        result.connection = connection.trimmingCharacters(in: .whitespacesAndNewlines)
        return result
    }
}

private struct Preferences: Codable {
    var servers: [ServerConfig]
    var refreshInterval: Double
}

private struct FleetTelemetry: Decodable {
    var timestamp: Double
    var servers: [TelemetryServer]
}

private struct TelemetryServer: Decodable, Identifiable {
    var name: String
    var connection: String?
    var address: String?
    var ok: Bool
    var error: String?
    var elapsed_ms: Double?
    var ssh_retries: Int?
    var _refresh_failures: Int?
    var _refresh_error: String?
    var gpus: [TelemetryGPU]?
    var id: String { name + "|" + (connection ?? "") }
}

private struct TelemetryGPU: Decodable, Identifiable {
    var index: Double?
    var name: String?
    var uuid: String?
    var error: String?
    var gpu_utilization: Double?
    var memory_total_bytes: Double?
    var memory_used_bytes: Double?
    var temperature: Double?
    var processes: [TelemetryProcess]?
    var id: String { uuid ?? "gpu-\(index ?? -1)" }
}

private struct TelemetryProcess: Decodable {
    var pid: Double?
}

private struct GPUHistoryPoint: Codable, Identifiable {
    var timestamp: Double
    var gpu: Double?
    var memory: Double?
    var temperature: Double?
    var id: Double { timestamp }
}

private struct SavedGPUHistory: Codable {
    var points: [String: [GPUHistoryPoint]]
}

private func intervalLabel(_ value: Double) -> String {
    value.formatted(.number.precision(.fractionLength(0...2))) + "s"
}

private enum PythonRuntime {
    // Finder-launched apps do not inherit the user's shell startup files.
    static var executable: String? {
        let shellPaths = (ProcessInfo.processInfo.environment["PATH"] ?? "")
            .split(separator: ":").map(String.init).filter { $0.hasPrefix("/") }
        let directories = ["/opt/homebrew/bin", "/usr/local/bin"] + shellPaths + ["/usr/bin"]
        return directories.map { URL(fileURLWithPath: $0).appendingPathComponent("python3").path }
            .first { FileManager.default.isExecutableFile(atPath: $0) }
    }
}

private enum ConnectionValidator {
    private struct Result: Decodable {
        var ok: Bool
        var error: String?
    }

    static func error(for server: ServerConfig) async -> String? {
        let server = server.cleaned
        if server.name.isEmpty { return "请填写服务器名称。" }
        if server.name.count > 80 { return "服务器名称最多 80 个字符。" }
        if server.name.contains("=") || server.name.hasPrefix("ssh ") {
            return "名称不能包含 =，或以 ssh 加空格开头。"
        }
        if server.connection.isEmpty { return "请填写 SSH 连接。" }
        if server.connection.contains("\n") || server.connection.contains("\r") {
            return "SSH 连接请填写在一行内。"
        }
        guard let script = Bundle.main.url(forResource: "gpu_watch", withExtension: "py") else {
            return "找不到内置监控脚本，请重新安装应用。"
        }
        guard let python = PythonRuntime.executable else {
            return "未找到 Python 3，请安装后重新打开应用。"
        }
        // The editor uses the collector's parser so both accept the same syntax.
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                let output = Pipe()
                let errors = Pipe()
                process.executableURL = URL(fileURLWithPath: python)
                process.arguments = [script.path, "--validate-connection", server.connection]
                process.standardOutput = output
                process.standardError = errors
                do {
                    try process.run()
                    let timeout = DispatchWorkItem {
                        if process.isRunning { process.terminate() }
                    }
                    let forceStop = DispatchWorkItem {
                        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
                    }
                    DispatchQueue.global().asyncAfter(deadline: .now() + 5, execute: timeout)
                    DispatchQueue.global().asyncAfter(deadline: .now() + 6, execute: forceStop)
                    defer { timeout.cancel(); forceStop.cancel() }
                    let data = output.fileHandleForReading.readDataToEndOfFile()
                    let errorData = errors.fileHandleForReading.readDataToEndOfFile()
                    process.waitUntilExit()
                    if process.terminationReason == .uncaughtSignal {
                        continuation.resume(returning: "本地校验未完成，请确认 Python 3 可用后重试。")
                    } else if let result = try? JSONDecoder().decode(Result.self, from: data) {
                        continuation.resume(returning: result.ok ? nil : result.error ?? "连接格式无效。")
                    } else {
                        let detail = String(data: errorData, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                        continuation.resume(returning: "无法校验连接，请确认 Python 3 可用。" + String(detail.prefix(160)))
                    }
                } catch {
                    continuation.resume(returning: "无法运行连接校验：\(error.localizedDescription)")
                }
            }
        }
    }
}

@MainActor
final class MonitorModel: ObservableObject {
    static let builtInServers = [
        ServerConfig(name: "GPU Server 01", connection: "gpu01.example.invalid:22"),
        ServerConfig(name: "GPU Server 02", connection: "gpu02.example.invalid:22"),
        ServerConfig(name: "GPU Server 03", connection: "gpu03.example.invalid:22"),
    ]

    @Published var servers: [ServerConfig] { didSet { save() } }
    @Published var refreshInterval: Double { didSet { save() } }
    @Published var notice = "配置已就绪"
    @Published var noticeIcon = "checkmark.circle"
    @Published var alertMessage: String?
    @Published private(set) var isLaunching = false
    @Published private(set) var isImporting = false
    @Published private var removedServer: RemovedServer?
    @Published fileprivate var telemetry: FleetTelemetry?
    @Published fileprivate var history: [String: [GPUHistoryPoint]] = [:]
    @Published var idleAlertsEnabled = UserDefaults.standard.bool(forKey: "gpuFleet.idleAlerts") {
        didSet { UserDefaults.standard.set(idleAlertsEnabled, forKey: "gpuFleet.idleAlerts") }
    }
    @Published var idleThresholdPercent = UserDefaults.standard.object(forKey: "gpuFleet.idleThreshold") == nil
        ? 5.0 : UserDefaults.standard.double(forKey: "gpuFleet.idleThreshold") {
        didSet { UserDefaults.standard.set(idleThresholdPercent, forKey: "gpuFleet.idleThreshold") }
    }
    @Published var idleDurationSeconds = UserDefaults.standard.object(forKey: "gpuFleet.idleDuration") == nil
        ? 60.0 : UserDefaults.standard.double(forKey: "gpuFleet.idleDuration") {
        didSet { UserDefaults.standard.set(idleDurationSeconds, forKey: "gpuFleet.idleDuration") }
    }

    private struct RemovedServer {
        let server: ServerConfig
        let index: Int
    }
    private let preferencesURL: URL
    private let telemetryURL: URL
    private let historyURL: URL
    private var needsPreferenceBackup = false
    private var lastTelemetryTimestamp: Double?
    private var lastHistoryWriteTimestamp: Double = 0
    private var idleSince: [String: Double] = [:]
    private var notifiedIdleGPUs = Set<String>()

    init() {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("GPU Fleet Monitor", isDirectory: true)
        preferencesURL = support.appendingPathComponent("preferences.json")
        telemetryURL = support.appendingPathComponent("live-telemetry.json")
        historyURL = support.appendingPathComponent("history.json")
        if let data = try? Data(contentsOf: historyURL),
           let saved = try? JSONDecoder().decode(SavedGPUHistory.self, from: data) {
            history = saved.points.mapValues { Array($0.suffix(360)) }
        }
        if let data = try? Data(contentsOf: preferencesURL),
           let saved = try? JSONDecoder().decode(Preferences.self, from: data) {
            // Old exported files may contain duplicate IDs; keep the rows usable.
            var seen = Set<UUID>()
            servers = saved.servers.map { value in
                var server = value
                if !seen.insert(server.id).inserted { server.id = UUID(); seen.insert(server.id) }
                return server
            }
            refreshInterval = saved.refreshInterval.isFinite ? min(60, max(0.25, saved.refreshInterval)) : 2
        } else {
            servers = Self.builtInServers
            refreshInterval = 2
            if FileManager.default.fileExists(atPath: preferencesURL.path) {
                needsPreferenceBackup = true
                notice = "原配置无法读取，将保留备份"
                noticeIcon = "exclamationmark.circle"
            }
        }
    }

    var enabledCount: Int { servers.filter(\.enabled).count }
    var canUndoRemoval: Bool { removedServer != nil }
    var latestTelemetryIsFresh: Bool {
        guard let telemetry else { return false }
        return Date().timeIntervalSince1970 - telemetry.timestamp < 12
    }

    func refreshTelemetry() {
        guard let data = try? Data(contentsOf: telemetryURL),
              let snapshot = try? JSONDecoder().decode(FleetTelemetry.self, from: data) else { return }
        telemetry = snapshot
        guard snapshot.timestamp != lastTelemetryTimestamp else { return }
        lastTelemetryTimestamp = snapshot.timestamp
        appendHistory(from: snapshot)
        evaluateIdleReminders(from: snapshot)
    }

    fileprivate func diagnosticDetail(for server: TelemetryServer) {
        let detail = server._refresh_error ?? server.error
        if let detail, !detail.isEmpty {
            alertMessage = "\(server.name)：\(detail)"
        } else {
            alertMessage = "\(server.name)：SSH 采集响应正常，最近耗时 \(Int(server.elapsed_ms ?? 0)) ms。"
        }
    }

    fileprivate func historyKey(for server: TelemetryServer, gpu: TelemetryGPU) -> String {
        server.id + "|" + gpu.id
    }

    func enableIdleReminders(_ enabled: Bool) {
        guard enabled else {
            idleAlertsEnabled = false
            idleSince.removeAll()
            notifiedIdleGPUs.removeAll()
            return
        }
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { [weak self] granted, _ in
            Task { @MainActor in
                guard let self else { return }
                if granted {
                    self.idleAlertsEnabled = true
                } else {
                    self.alertMessage = "系统未授权通知，请在系统设置中允许 GPU Fleet Monitor 通知。"
                }
            }
        }
    }

    private func appendHistory(from snapshot: FleetTelemetry) {
        guard Date().timeIntervalSince1970 - snapshot.timestamp < 12,
              snapshot.timestamp - lastHistoryWriteTimestamp >= 5 else { return }
        var nextHistory = history
        for server in snapshot.servers where server.ok && (server._refresh_failures ?? 0) == 0 {
            for gpu in server.gpus ?? [] where gpu.error == nil {
                let key = historyKey(for: server, gpu: gpu)
                let memoryPercent: Double? = {
                    guard let used = gpu.memory_used_bytes, let total = gpu.memory_total_bytes, total > 0 else { return nil }
                    return min(100, max(0, used / total * 100))
                }()
                let point = GPUHistoryPoint(timestamp: snapshot.timestamp,
                                            gpu: gpu.gpu_utilization,
                                            memory: memoryPercent,
                                            temperature: gpu.temperature)
                nextHistory[key, default: []].append(point)
                if nextHistory[key, default: []].count > 360 {
                    nextHistory[key] = Array(nextHistory[key, default: []].suffix(360))
                }
            }
        }
        history = nextHistory
        lastHistoryWriteTimestamp = snapshot.timestamp
        do {
            try FileManager.default.createDirectory(at: historyURL.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: historyURL.deletingLastPathComponent().path)
            let data = try JSONEncoder().encode(SavedGPUHistory(points: nextHistory))
            try data.write(to: historyURL, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: historyURL.path)
        } catch {
            notice = "历史曲线暂时无法保存"
            noticeIcon = "exclamationmark.circle"
        }
    }

    private func evaluateIdleReminders(from snapshot: FleetTelemetry) {
        guard idleAlertsEnabled, Date().timeIntervalSince1970 - snapshot.timestamp < 12 else { return }
        var visibleGPUs = Set<String>()
        for server in snapshot.servers where server.ok && (server._refresh_failures ?? 0) == 0 {
            for gpu in server.gpus ?? [] where gpu.error == nil {
                let key = historyKey(for: server, gpu: gpu)
                visibleGPUs.insert(key)
                let noProcesses = gpu.processes?.isEmpty == true
                let isIdle = noProcesses && (gpu.gpu_utilization ?? 100) <= idleThresholdPercent
                if isIdle {
                    let began = idleSince[key] ?? snapshot.timestamp
                    idleSince[key] = began
                    if snapshot.timestamp - began >= idleDurationSeconds && !notifiedIdleGPUs.contains(key) {
                        notifiedIdleGPUs.insert(key)
                        let index = gpu.index.map { String(Int($0)) } ?? "?"
                        let content = UNMutableNotificationContent()
                        content.title = "GPU 空闲提醒"
                        content.body = "\(server.name) · GPU \(index) 连续空闲超过 \(Int(idleDurationSeconds)) 秒。"
                        content.sound = .default
                        let request = UNNotificationRequest(identifier: "gpuFleet.idle.\(UUID().uuidString)", content: content, trigger: nil)
                        UNUserNotificationCenter.current().add(request)
                    }
                } else {
                    idleSince.removeValue(forKey: key)
                    notifiedIdleGPUs.remove(key)
                }
            }
        }
        idleSince = idleSince.filter { visibleGPUs.contains($0.key) }
    }

    func commit(_ server: ServerConfig) {
        if let index = servers.firstIndex(where: { $0.id == server.id }) {
            servers[index] = server.cleaned
        } else {
            servers.append(server.cleaned)
        }
    }

    func toggleServer(id: UUID) {
        guard let index = servers.firstIndex(where: { $0.id == id }) else { return }
        servers[index].enabled.toggle()
    }

    func setAllEnabled(_ enabled: Bool) {
        servers = servers.map { value in
            var server = value
            server.enabled = enabled
            return server
        }
    }

    func moveServer(id: UUID, by offset: Int) {
        guard let index = servers.firstIndex(where: { $0.id == id }),
              servers.indices.contains(index + offset) else { return }
        servers.swapAt(index, index + offset)
    }

    func removeServer(id: UUID) {
        guard let index = servers.firstIndex(where: { $0.id == id }) else { return }
        removedServer = RemovedServer(server: servers[index], index: index)
        servers.remove(at: index)
        notice = "已移除服务器"
        noticeIcon = "minus.circle"
    }

    func undoRemoval() {
        guard let removed = removedServer else { return }
        if !servers.contains(where: { $0.id == removed.server.id }) {
            servers.insert(removed.server, at: min(removed.index, servers.count))
        }
        removedServer = nil
        notice = "已恢复服务器"
        noticeIcon = "arrow.uturn.backward"
    }

    func addMissingBuiltIns() {
        let existing = Set(servers.map { $0.connection.trimmingCharacters(in: .whitespacesAndNewlines) })
        let added = Self.builtInServers.filter { !existing.contains($0.connection) }.map { value in
            var server = value
            server.id = UUID()
            return server
        }
        servers.append(contentsOf: added)
        notice = added.isEmpty ? "内置服务器已齐全" : "已补充 \(added.count) 台服务器"
        noticeIcon = "server.rack"
    }

    func copyConnection(_ server: ServerConfig) {
        copyText(server.connection)
        notice = "已复制连接"
        noticeIcon = "doc.on.doc"
    }

    func copyMonitorCommand() {
        guard let command = invocation(for: servers.filter(\.enabled)) else { return }
        copyText(command)
        notice = "已复制监控命令"
        noticeIcon = "doc.on.doc"
    }

    func exportConfiguration() {
        let panel = NSSavePanel()
        panel.title = "导出配置"
        panel.nameFieldStringValue = "GPU Fleet.json"
        panel.allowedContentTypes = [.json]
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(Preferences(servers: servers, refreshInterval: refreshInterval)).write(to: url, options: .atomic)
            notice = "配置已导出"
            noticeIcon = "square.and.arrow.up"
        } catch { alertMessage = "导出失败：\(error.localizedDescription)" }
    }

    func importConfiguration() {
        let panel = NSOpenPanel()
        panel.title = "导入服务器"
        panel.message = "合并到现有列表，跳过重复连接。"
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            guard size <= 512_000 else {
                alertMessage = "配置文件过大，请使用本应用导出的 JSON 文件。"
                return
            }
            let imported = try JSONDecoder().decode(Preferences.self, from: Data(contentsOf: url))
            guard imported.servers.count <= 100 else {
                alertMessage = "一次最多导入 100 台服务器。"
                return
            }
            isImporting = true
            notice = "正在校验导入配置…"
            noticeIcon = "arrow.down.doc"
            Task {
                var added: [ServerConfig] = []
                for item in imported.servers {
                    var server = item.cleaned
                    if let error = await ConnectionValidator.error(for: server) {
                        isImporting = false
                        notice = "导入未完成"
                        alertMessage = "\(server.name)：\(error)"
                        return
                    }
                    server.id = UUID()
                    added.append(server)
                }
                // Read current state after validation in case the list changed.
                var connections = Set(servers.map { $0.cleaned.connection })
                let unique = added.filter { connections.insert($0.connection).inserted }
                servers.append(contentsOf: unique)
                isImporting = false
                notice = unique.isEmpty ? "这些连接已在列表中" : "已导入 \(unique.count) 台服务器"
                noticeIcon = "checkmark.circle"
            }
        } catch { alertMessage = "无法导入，请选择本应用导出的配置：\(error.localizedDescription)" }
    }

    func launchMonitor(only id: UUID? = nil) {
        guard !isLaunching else { return }
        let selected = id.map { id in servers.filter { $0.id == id } } ?? servers.filter(\.enabled)
        guard let command = invocation(for: selected, telemetryFile: telemetryURL.path) else { return }
        guard let terminalURL = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Terminal") else {
            alertMessage = "找不到 macOS Terminal.app。"
            return
        }
        do {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("GPU Fleet Monitor", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
            let commandURL = directory.appendingPathComponent("GPU-Monitor-\(UUID().uuidString).command")
            let script = """
            #!/bin/zsh
            trap 'rm -f -- "$0"' EXIT
            \(command)
            printf '\\n监控已结束。按回车关闭此终端会话。\\n'
            read
            """
            try Data(script.utf8).write(to: commandURL, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: commandURL.path)
            isLaunching = true
            NSWorkspace.shared.open([commandURL], withApplicationAt: terminalURL, configuration: NSWorkspace.OpenConfiguration()) { [weak self] _, error in
                Task { @MainActor in
                    self?.isLaunching = false
                    if let error {
                        self?.alertMessage = "无法打开 Terminal：\(error.localizedDescription)"
                    } else {
                        self?.notice = "已在 Terminal 打开监控"
                        self?.noticeIcon = "terminal"
                    }
                }
            }
        } catch { alertMessage = "启动失败：\(error.localizedDescription)" }
    }

    private func invocation(for selected: [ServerConfig], telemetryFile: String? = nil) -> String? {
        guard !selected.isEmpty else { alertMessage = "请先选择服务器。"; return nil }
        guard selected.allSatisfy({ !$0.cleaned.name.isEmpty && !$0.cleaned.connection.isEmpty }) else {
            alertMessage = "请先补全服务器名称和连接。"
            return nil
        }
        guard let scriptURL = Bundle.main.url(forResource: "gpu_watch", withExtension: "py") else {
            alertMessage = "找不到内置监控脚本，请重新安装应用。"
            return nil
        }
        guard let python = PythonRuntime.executable else {
            alertMessage = "未找到 Python 3，请安装后重新打开应用。"
            return nil
        }
        var arguments = [shellQuote(python), shellQuote(scriptURL.path), "--interval", String(refreshInterval)]
        if let telemetryFile { arguments += ["--telemetry-file", shellQuote(telemetryFile)] }
        arguments += selected.flatMap { ["--server", shellQuote("\($0.name)=\($0.connection)")] }
        return arguments.joined(separator: " ")
    }

    private func save() {
        do {
            try FileManager.default.createDirectory(at: preferencesURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            if needsPreferenceBackup {
                let backup = preferencesURL.deletingLastPathComponent().appendingPathComponent("preferences-backup-\(UUID().uuidString).json")
                try FileManager.default.copyItem(at: preferencesURL, to: backup)
                needsPreferenceBackup = false
            }
            try JSONEncoder().encode(Preferences(servers: servers, refreshInterval: refreshInterval)).write(to: preferencesURL, options: .atomic)
            notice = "已自动保存"
            noticeIcon = "checkmark.circle"
        } catch {
            notice = "配置未保存"
            noticeIcon = "exclamationmark.circle"
            alertMessage = "无法保存配置：\(error.localizedDescription)"
        }
    }

    private func copyText(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }

    private func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}

@main
struct GPUFleetApp: App {
    @StateObject private var model = MonitorModel()

    var body: some Scene {
        WindowGroup("GPU Fleet") {
            ContentView()
                .environmentObject(model)
                .frame(minWidth: 680, minHeight: 460)
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.automatic)
        .defaultSize(width: 760, height: 520)
        .commands { CommandGroup(replacing: .newItem) {} }
    }
}

private enum FleetPalette {
    static let accent = Color(red: 0.20, green: 0.48, blue: 0.94)
    static let outline = Color.primary.opacity(0.07)
}

private struct GlassButton: ViewModifier {
    var prominent = false
    @ViewBuilder
    func body(content: Content) -> some View {
        if #available(macOS 26.0, *) {
            if prominent { content.buttonStyle(.glassProminent) }
            else { content.buttonStyle(.glass) }
        } else {
            if prominent { content.buttonStyle(.borderedProminent) }
            else { content.buttonStyle(.bordered) }
        }
    }
}

private struct GlassCapsule: ViewModifier {
    @ViewBuilder
    func body(content: Content) -> some View {
        if #available(macOS 26.0, *) {
            content.glassEffect(.regular, in: Capsule())
        } else {
            content.background(.thinMaterial, in: Capsule())
                .overlay(Capsule().strokeBorder(FleetPalette.outline, lineWidth: 1))
        }
    }
}

struct ContentView: View {
    @EnvironmentObject private var model: MonitorModel
    @Environment(\.colorScheme) private var colorScheme
    @State private var search = ""
    @State private var editingServer: ServerConfig?
    @State private var customIntervalOpen = false
    @State private var insightsOpen = false
    private let presets: [Double] = [0.25, 0.5, 1, 2, 5]

    private var filteredServers: [ServerConfig] {
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
        return query.isEmpty ? model.servers : model.servers.filter {
            $0.name.localizedCaseInsensitiveContains(query) || $0.connection.localizedCaseInsensitiveContains(query)
        }
    }

    var body: some View {
        ZStack {
            Color(nsColor: .windowBackgroundColor).ignoresSafeArea()
            LinearGradient(colors: [FleetPalette.accent.opacity(colorScheme == .dark ? 0.10 : 0.045), .clear, Color.cyan.opacity(0.035)], startPoint: .topLeading, endPoint: .bottomTrailing)
                .ignoresSafeArea().allowsHitTesting(false)
            VStack(spacing: 16) {
                header
                serverList
                refreshControls
                footer
            }
            .padding(.horizontal, 24)
            .padding(.top, 22)
            .padding(.bottom, 20)
            .frame(maxWidth: 940)
        }
        .tint(FleetPalette.accent)
        .sheet(item: $editingServer) { server in
            ServerEditorSheet(server: server, isNew: !model.servers.contains { $0.id == server.id }) { model.commit($0) }
        }
        .sheet(isPresented: $insightsOpen) {
            MonitorInsightsSheet().environmentObject(model)
        }
        .onReceive(Timer.publish(every: 1, on: .main, in: .common).autoconnect()) { _ in
            model.refreshTelemetry()
        }
        .alert("GPU Fleet", isPresented: Binding(get: { model.alertMessage != nil }, set: { if !$0 { model.alertMessage = nil } })) {
            Button("好", role: .cancel) { model.alertMessage = nil }
        } message: { Text(model.alertMessage ?? "") }
    }

    private var header: some View {
        HStack(spacing: 11) {
            Group {
                if let url = Bundle.main.url(forResource: "GPUFleetAppIcon", withExtension: "png"), let icon = NSImage(contentsOf: url) {
                    Image(nsImage: icon).resizable().scaledToFill()
                } else {
                    Image(systemName: "cpu").resizable().scaledToFit().padding(9)
                }
            }
            .frame(width: 38, height: 38)
            .clipShape(RoundedRectangle(cornerRadius: 11, style: .continuous))
            VStack(alignment: .leading, spacing: 3) {
                Text("GPU Fleet").font(.system(size: 21, weight: .semibold, design: .rounded)).tracking(-0.4)
                Text("远程显卡监控").font(.system(size: 11)).foregroundStyle(.secondary)
            }
            Spacer()
            Button { insightsOpen = true } label: {
                Image(systemName: "chart.xyaxis.line")
                    .font(.system(size: 14, weight: .medium))
                    .frame(width: 28, height: 30)
            }
            .buttonStyle(.borderless)
            .help("连接诊断、空闲提醒与历史曲线")
            .accessibilityLabel("连接诊断与历史曲线")
            Menu {
                Button("导入服务器…", systemImage: "square.and.arrow.down", action: model.importConfiguration).disabled(model.isImporting)
                Button("导出配置…", systemImage: "square.and.arrow.up", action: model.exportConfiguration)
                Button("复制监控命令", systemImage: "doc.on.doc", action: model.copyMonitorCommand).disabled(model.enabledCount == 0)
                Divider()
                Button("选择全部") { model.setAllEnabled(true) }
                Button("取消选择") { model.setAllEnabled(false) }
                Divider()
                Button("补充内置服务器", systemImage: "server.rack", action: model.addMissingBuiltIns)
            } label: {
                Image(systemName: "ellipsis").font(.system(size: 16, weight: .medium)).frame(width: 26, height: 30)
            }
            .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
            .help("配置与批量操作").accessibilityLabel("配置与批量操作")
            Button { editingServer = ServerConfig(name: "", connection: "") } label: {
                Label("添加", systemImage: "plus").font(.system(size: 12, weight: .medium)).padding(.horizontal, 3)
            }
            .modifier(GlassButton()).controlSize(.regular)
            .keyboardShortcut("n", modifiers: .command)
        }
        .padding(.bottom, 2)
    }

    private var serverList: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Text("服务器").font(.system(size: 12, weight: .semibold))
                Text("\(model.enabledCount) / \(model.servers.count) 已选")
                    .font(.system(size: 11).monospacedDigit()).foregroundStyle(.secondary)
                Spacer()
                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                    TextField("搜索名称或地址", text: $search).textFieldStyle(.plain)
                        .accessibilityLabel("搜索服务器")
                    if !search.isEmpty {
                        Button { search = "" } label: { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary) }
                            .buttonStyle(.plain).accessibilityLabel("清除搜索")
                    }
                }
                .font(.system(size: 11))
                .padding(.horizontal, 9).padding(.vertical, 6)
                .frame(width: 188)
                .background(Color.primary.opacity(0.035), in: Capsule())
            }
            .padding(.horizontal, 16).padding(.vertical, 12)
            Divider().opacity(0.6)
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(filteredServers) { server in
                        ServerRow(server: server,
                                  isFirst: model.servers.first?.id == server.id,
                                  isLast: model.servers.last?.id == server.id,
                                  onToggle: { model.toggleServer(id: server.id) },
                                  onEdit: { editingServer = server },
                                  onDuplicate: { editingServer = ServerConfig(name: server.name + " 副本", connection: server.connection, enabled: server.enabled) },
                                  onCopy: { model.copyConnection(server) },
                                  onLaunch: { model.launchMonitor(only: server.id) },
                                  onMove: { model.moveServer(id: server.id, by: $0) },
                                  onDelete: { model.removeServer(id: server.id) })
                        if server.id != filteredServers.last?.id {
                            Divider().padding(.leading, 68).padding(.trailing, 16).opacity(0.55)
                        }
                    }
                    if filteredServers.isEmpty {
                        VStack(spacing: 9) {
                            Image(systemName: search.isEmpty ? "server.rack" : "magnifyingglass")
                                .font(.system(size: 23, weight: .light)).foregroundStyle(.tertiary)
                            Text(search.isEmpty ? "添加第一台服务器" : "没有匹配的服务器")
                                .font(.system(size: 12)).foregroundStyle(.secondary)
                            if search.isEmpty {
                                Button("添加服务器") { editingServer = ServerConfig(name: "", connection: "") }.buttonStyle(.borderless)
                            }
                        }
                        .frame(maxWidth: .infinity).padding(.vertical, 28)
                    }
                }
            }
            .frame(maxHeight: .infinity)
        }
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 19, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 19, style: .continuous).strokeBorder(FleetPalette.outline, lineWidth: 1))
        .shadow(color: .black.opacity(colorScheme == .dark ? 0.1 : 0.035), radius: 14, y: 6)
    }

    private var refreshControls: some View {
        HStack(spacing: 10) {
            Image(systemName: "arrow.triangle.2.circlepath").font(.system(size: 12, weight: .medium)).foregroundStyle(.secondary)
            Text("刷新").font(.system(size: 12, weight: .medium)).foregroundStyle(.secondary)
            Spacer(minLength: 4)
            HStack(spacing: 2) {
                ForEach(presets, id: \.self) { value in
                    Button {
                        withAnimation(.easeInOut(duration: 0.16)) { model.refreshInterval = value }
                    } label: {
                        Text(intervalLabel(value)).font(.system(size: 11, weight: .medium).monospacedDigit())
                            .foregroundStyle(model.refreshInterval == value ? Color.white : Color.primary.opacity(0.72))
                            .frame(minWidth: 38).padding(.horizontal, 6).padding(.vertical, 7)
                            .background(model.refreshInterval == value ? FleetPalette.accent : Color.clear, in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .help(value == 0.25 ? "高刷新 · 0.25 秒" : "每 \(intervalLabel(value)) 采集一次")
                    .accessibilityLabel("刷新间隔 \(intervalLabel(value))")
                    .accessibilityAddTraits(model.refreshInterval == value ? .isSelected : [])
                }
            }
            .padding(3).modifier(GlassCapsule())
            Button { customIntervalOpen = true } label: {
                HStack(spacing: 4) {
                    Text(presets.contains(model.refreshInterval) ? "自定" : intervalLabel(model.refreshInterval))
                    Image(systemName: "chevron.down").font(.system(size: 8, weight: .semibold))
                }.font(.system(size: 11, weight: .medium)).frame(minWidth: 44)
            }
            .buttonStyle(.borderless).help("自定义刷新间隔")
            .popover(isPresented: $customIntervalOpen, arrowEdge: .bottom) {
                IntervalEditor(interval: model.refreshInterval) { value in
                    model.refreshInterval = value
                    customIntervalOpen = false
                }
            }
        }
        .padding(.horizontal, 4)
    }

    private var footer: some View {
        HStack(spacing: 7) {
            if model.isImporting { ProgressView().controlSize(.small) }
            else { Image(systemName: model.noticeIcon).font(.system(size: 12)).foregroundStyle(.secondary) }
            Text(model.notice).font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(1)
            if model.canUndoRemoval {
                Button("撤销", action: model.undoRemoval).buttonStyle(.borderless).font(.system(size: 11, weight: .medium))
            }
            Spacer(minLength: 8)
            Button { model.launchMonitor() } label: {
                HStack(spacing: 8) {
                    Image(systemName: "terminal").font(.system(size: 13, weight: .semibold))
                    Text(model.isLaunching ? "正在打开…" : "启动监控").font(.system(size: 13, weight: .semibold))
                    Text("⌘↩").font(.system(size: 11, weight: .medium)).opacity(0.65)
                }
                .padding(.horizontal, 8).padding(.vertical, 3)
            }
            .modifier(GlassButton(prominent: true)).controlSize(.large)
            .keyboardShortcut(.return, modifiers: .command)
            .disabled(model.enabledCount == 0 || model.isLaunching)
        }
        .padding(.top, 2)
    }
}

private struct GPUChoice: Identifiable {
    var id: String
    var label: String
}

private enum HistoryMetric: String, CaseIterable, Identifiable {
    case gpu, memory, temperature
    var id: String { rawValue }
    var title: String {
        switch self {
        case .gpu: "GPU"
        case .memory: "显存"
        case .temperature: "温度"
        }
    }
    var unit: String { self == .temperature ? "°C" : "%" }
    var limit: ClosedRange<Double> { self == .temperature ? 20...100 : 0...100 }
    var color: Color {
        switch self {
        case .gpu: Color(red: 0.25, green: 0.58, blue: 0.96)
        case .memory: Color(red: 0.40, green: 0.72, blue: 0.56)
        case .temperature: Color(red: 0.91, green: 0.60, blue: 0.31)
        }
    }
    func value(_ point: GPUHistoryPoint) -> Double? {
        switch self {
        case .gpu: point.gpu
        case .memory: point.memory
        case .temperature: point.temperature
        }
    }
}

private struct MonitorInsightsSheet: View {
    @EnvironmentObject private var model: MonitorModel
    @Environment(\.dismiss) private var dismiss
    @State private var selectedGPUKey: String?

    private var choices: [GPUChoice] {
        guard let telemetry = model.telemetry else { return [] }
        return telemetry.servers.flatMap { server in
            (server.gpus ?? []).compactMap { gpu in
                guard gpu.error == nil else { return nil }
                let key = model.historyKey(for: server, gpu: gpu)
                let index = gpu.index.map { String(Int($0)) } ?? "?"
                return GPUChoice(id: key, label: "\(server.name) · GPU \(index)")
            }
        }
    }

    private var selectedHistory: [GPUHistoryPoint] {
        guard let key = selectedGPUKey ?? choices.first?.id else { return [] }
        return model.history[key] ?? []
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Image(systemName: "waveform.path.ecg").font(.system(size: 17, weight: .medium)).foregroundStyle(FleetPalette.accent)
                    .frame(width: 34, height: 34).background(FleetPalette.accent.opacity(0.09), in: RoundedRectangle(cornerRadius: 10))
                VStack(alignment: .leading, spacing: 2) {
                    Text("监控洞察").font(.system(size: 16, weight: .semibold))
                    Text("连接诊断 · 空闲提醒 · 近 30 分钟趋势").font(.system(size: 10)).foregroundStyle(.secondary)
                }
                Spacer()
                Button { dismiss() } label: { Image(systemName: "xmark").font(.system(size: 11, weight: .semibold)).frame(width: 28, height: 28) }
                    .buttonStyle(.borderless).help("关闭")
            }

            connectionCard
            idleCard
            historyCard
        }
        .padding(18)
        .frame(width: 680, height: 465)
        .background(Color(nsColor: .windowBackgroundColor))
        .tint(FleetPalette.accent)
        .onAppear { if selectedGPUKey == nil { selectedGPUKey = choices.first?.id } }
        .onChange(of: choices.map(\.id)) { values in
            if let selectedGPUKey, values.contains(selectedGPUKey) { return }
            selectedGPUKey = values.first
        }
    }

    private var connectionCard: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Label("连接诊断", systemImage: "network")
                    .font(.system(size: 11, weight: .semibold))
                Spacer()
                Text(model.latestTelemetryIsFresh ? "实时" : "等待新采样")
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(model.latestTelemetryIsFresh ? .green : .secondary)
            }
            if let telemetry = model.telemetry {
                ForEach(telemetry.servers) { server in
                    Button { model.diagnosticDetail(for: server) } label: {
                        HStack(spacing: 9) {
                            Circle().fill(statusColor(for: server)).frame(width: 6, height: 6)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(server.name).font(.system(size: 10, weight: .medium)).lineLimit(1)
                                Text(server.address ?? server.connection ?? "")
                                    .font(.system(size: 9, design: .monospaced)).foregroundStyle(.secondary).lineLimit(1)
                            }
                            Spacer(minLength: 4)
                            Text(statusLabel(for: server)).font(.system(size: 9, weight: .medium)).foregroundStyle(statusColor(for: server))
                            if let elapsed = server.elapsed_ms {
                                Text("\(Int(elapsed)) ms").font(.system(size: 9, design: .monospaced)).foregroundStyle(.secondary).frame(width: 55, alignment: .trailing)
                            }
                            if (server.ssh_retries ?? 0) > 0 || (server._refresh_failures ?? 0) > 0 {
                                Image(systemName: "arrow.clockwise").font(.system(size: 9, weight: .medium)).foregroundStyle(.orange)
                            }
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            } else {
                Text("启动监控后显示 SSH 状态、响应耗时与错误原因。")
                    .font(.system(size: 10)).foregroundStyle(.secondary).padding(.vertical, 3)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 13, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 13, style: .continuous).strokeBorder(FleetPalette.outline, lineWidth: 1))
    }

    private var idleCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Toggle(isOn: Binding(get: { model.idleAlertsEnabled }, set: model.enableIdleReminders)) {
                    Label("GPU 空闲提醒", systemImage: "bell.badge")
                        .font(.system(size: 11, weight: .semibold))
                }
                .toggleStyle(.switch).controlSize(.mini)
                Spacer(minLength: 4)
                Menu {
                    ForEach([5.0, 10, 20], id: \.self) { value in
                        Button("GPU ≤ \(Int(value))%") { model.idleThresholdPercent = value }
                    }
                } label: {
                    Text("≤ \(Int(model.idleThresholdPercent))%")
                        .font(.system(size: 9, weight: .medium)).padding(.horizontal, 8).padding(.vertical, 5)
                        .background(Color.primary.opacity(0.045), in: Capsule())
                }
                .menuStyle(.borderlessButton).disabled(!model.idleAlertsEnabled)
                Menu {
                    ForEach([30.0, 60, 300], id: \.self) { seconds in
                        Button(seconds == 30 ? "30 秒" : "\(Int(seconds / 60)) 分钟") { model.idleDurationSeconds = seconds }
                    }
                } label: {
                    Text(model.idleDurationSeconds >= 60 ? "\(Int(model.idleDurationSeconds / 60)) 分钟" : "\(Int(model.idleDurationSeconds)) 秒")
                        .font(.system(size: 9, weight: .medium)).padding(.horizontal, 8).padding(.vertical, 5)
                        .background(Color.primary.opacity(0.045), in: Capsule())
                }
                .menuStyle(.borderlessButton).disabled(!model.idleAlertsEnabled)
            }
            Text("无运行进程且 GPU 使用率低于阈值时通知；需要允许系统通知。")
                .font(.system(size: 9)).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 13, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 13, style: .continuous).strokeBorder(FleetPalette.outline, lineWidth: 1))
    }

    private var historyCard: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 6) {
                Label("历史曲线", systemImage: "chart.xyaxis.line").font(.system(size: 11, weight: .semibold))
                Spacer()
                if !choices.isEmpty {
                    Picker("显卡", selection: Binding(get: { selectedGPUKey ?? choices[0].id }, set: { selectedGPUKey = $0 })) {
                        ForEach(choices) { choice in Text(choice.label).tag(choice.id) }
                    }
                    .labelsHidden().pickerStyle(.menu).controlSize(.small).frame(maxWidth: 210)
                }
                Text("30 分钟").font(.system(size: 9)).foregroundStyle(.secondary)
            }
            if choices.isEmpty {
                VStack(spacing: 6) {
                    Image(systemName: "chart.xyaxis.line").font(.system(size: 20, weight: .light)).foregroundStyle(.tertiary)
                    Text("采样后显示 GPU、显存与温度趋势")
                        .font(.system(size: 10)).foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                HStack(spacing: 7) {
                    ForEach(HistoryMetric.allCases) { metric in
                        MetricTrendCard(metric: metric, points: selectedHistory)
                    }
                }
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 13, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 13, style: .continuous).strokeBorder(FleetPalette.outline, lineWidth: 1))
    }

    private func statusColor(for server: TelemetryServer) -> Color {
        guard model.latestTelemetryIsFresh else { return .secondary }
        if server.ok && (server._refresh_failures ?? 0) == 0 { return .green }
        return server.ok ? .orange : .red
    }

    private func statusLabel(for server: TelemetryServer) -> String {
        guard model.latestTelemetryIsFresh else { return "过期" }
        if server.ok && (server._refresh_failures ?? 0) == 0 { return "在线" }
        return server.ok ? "重连中" : "离线"
    }
}

private struct MetricTrendCard: View {
    let metric: HistoryMetric
    let points: [GPUHistoryPoint]

    private var latestValue: Double? {
        points.reversed().compactMap(metric.value).first
    }

    private var validPoints: [GPUHistoryPoint] {
        points.filter { metric.value($0) != nil }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(metric.title).font(.system(size: 9, weight: .medium)).foregroundStyle(.secondary)
                Spacer(minLength: 0)
                if let latestValue {
                    Text("\(Int(latestValue.rounded()))\(metric.unit)")
                        .font(.system(size: 10, weight: .semibold, design: .rounded).monospacedDigit())
                        .foregroundStyle(metric.color)
                } else {
                    Text("—").font(.system(size: 10)).foregroundStyle(.tertiary)
                }
            }
            Chart {
                ForEach(validPoints) { point in
                    if let value = metric.value(point) {
                        AreaMark(x: .value("时间", Date(timeIntervalSince1970: point.timestamp)),
                                 yStart: .value("基线", metric.limit.lowerBound),
                                 yEnd: .value(metric.title, value))
                            .foregroundStyle(metric.color.opacity(0.12).gradient)
                        LineMark(x: .value("时间", Date(timeIntervalSince1970: point.timestamp)),
                                 y: .value(metric.title, value))
                            .foregroundStyle(metric.color)
                            .lineStyle(StrokeStyle(lineWidth: 1.7, lineCap: .round, lineJoin: .round))
                    }
                }
            }
            .chartYScale(domain: metric.limit)
            .chartXAxis(.hidden).chartYAxis(.hidden)
            .frame(height: 58)
            if validPoints.isEmpty {
                Text("等待采样").font(.system(size: 8)).foregroundStyle(.tertiary)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(metric.color.opacity(0.045), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

private struct ServerRow: View {
    let server: ServerConfig
    let isFirst: Bool
    let isLast: Bool
    let onToggle: () -> Void
    let onEdit: () -> Void
    let onDuplicate: () -> Void
    let onCopy: () -> Void
    let onLaunch: () -> Void
    let onMove: (Int) -> Void
    let onDelete: () -> Void
    @State private var hovered = false

    var body: some View {
        HStack(spacing: 12) {
            Button(action: onToggle) {
                Image(systemName: server.enabled ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 18, weight: .regular))
                    .foregroundStyle(server.enabled ? FleetPalette.accent : Color.secondary.opacity(0.45))
                    .frame(width: 22, height: 34)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("\(server.enabled ? "取消选择" : "选择") \(server.name)")
            Button(action: onEdit) {
                HStack(spacing: 11) {
                    Image(systemName: "server.rack")
                        .font(.system(size: 16, weight: .regular))
                        .foregroundStyle(server.enabled ? FleetPalette.accent : .secondary)
                        .frame(width: 34, height: 34)
                        .background(FleetPalette.accent.opacity(server.enabled ? 0.075 : 0.025), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    VStack(alignment: .leading, spacing: 5) {
                        Text(server.name).font(.system(size: 13, weight: .semibold)).foregroundStyle(.primary).lineLimit(1)
                        Text(server.connection).font(.system(size: 10.5, design: .monospaced)).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain).accessibilityLabel("编辑 \(server.name)")
            .help(server.connection)
            Button(action: onEdit) {
                Image(systemName: "pencil").font(.system(size: 11, weight: .medium)).frame(width: 23, height: 28)
            }
            .buttonStyle(.borderless).foregroundStyle(.secondary)
            .opacity(hovered ? 1 : 0.45).help("编辑服务器")
            Menu {
                Button("单独启动", systemImage: "terminal", action: onLaunch)
                Button("编辑…", systemImage: "pencil", action: onEdit)
                Button("复制连接", systemImage: "doc.on.doc", action: onCopy)
                Button("创建副本…", systemImage: "plus.square.on.square", action: onDuplicate)
                Divider()
                Button("上移", systemImage: "arrow.up") { onMove(-1) }.disabled(isFirst)
                Button("下移", systemImage: "arrow.down") { onMove(1) }.disabled(isLast)
                Divider()
                Button("移除", systemImage: "trash", role: .destructive, action: onDelete)
            } label: {
                Image(systemName: "ellipsis").font(.system(size: 14, weight: .medium)).frame(width: 20, height: 28)
            }
            .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
            .help("\(server.name) 的操作").accessibilityLabel("\(server.name) 的操作")
        }
        .padding(.horizontal, 15).padding(.vertical, 12)
        .background(hovered ? FleetPalette.accent.opacity(0.035) : .clear)
        .opacity(server.enabled ? 1 : 0.58)
        .onHover { hovered = $0 }
    }
}

private struct ServerEditorSheet: View {
    @Environment(\.dismiss) private var dismiss
    @State private var draft: ServerConfig
    @State private var validationError: String?
    @State private var saving = false
    @FocusState private var focusName: Bool
    let isNew: Bool
    let onSave: (ServerConfig) -> Void

    init(server: ServerConfig, isNew: Bool, onSave: @escaping (ServerConfig) -> Void) {
        _draft = State(initialValue: server)
        self.isNew = isNew
        self.onSave = onSave
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack(spacing: 10) {
                Image(systemName: "server.rack").font(.system(size: 18)).foregroundStyle(FleetPalette.accent)
                    .frame(width: 38, height: 38).background(FleetPalette.accent.opacity(0.09), in: RoundedRectangle(cornerRadius: 12))
                Text(isNew ? "添加服务器" : "编辑服务器").font(.system(size: 18, weight: .semibold))
                Spacer()
            }
            VStack(alignment: .leading, spacing: 7) {
                Text("名称").font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary)
                TextField("例如：A100 · 实验室", text: $draft.name)
                    .textFieldStyle(.plain).font(.system(size: 13)).focused($focusName)
                    .disabled(saving)
                    .padding(11).background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 10))
            }
            VStack(alignment: .leading, spacing: 7) {
                Text("SSH 连接").font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary)
                TextField("user@host:port 或 ssh -p 端口 user@host", text: $draft.connection)
                    .textFieldStyle(.plain).font(.system(size: 12, design: .monospaced))
                    .disabled(saving)
                    .autocorrectionDisabled().padding(11)
                    .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 10))
                if let validationError {
                    Label(validationError, systemImage: "exclamationmark.circle")
                        .font(.system(size: 11)).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
                }
            }
            Toggle("纳入监控", isOn: $draft.enabled).toggleStyle(.switch).controlSize(.small).font(.system(size: 12)).disabled(saving)
            HStack(spacing: 10) {
                Spacer()
                if saving { ProgressView().controlSize(.small) }
                Button("取消") { dismiss() }.keyboardShortcut(.cancelAction).disabled(saving)
                Button(isNew ? "添加" : "保存", action: submit)
                    .modifier(GlassButton(prominent: true)).keyboardShortcut(.defaultAction)
                    .disabled(saving || draft.cleaned.name.isEmpty || draft.cleaned.connection.isEmpty)
            }
        }
        .padding(26).frame(width: 490)
        .tint(FleetPalette.accent)
        .onAppear { focusName = true }
        .onChange(of: draft.connection) { _ in validationError = nil }
        .onChange(of: draft.name) { _ in validationError = nil }
    }

    private func submit() {
        saving = true
        Task {
            let candidate = draft.cleaned
            if let error = await ConnectionValidator.error(for: candidate) {
                validationError = error
            } else {
                onSave(candidate)
                dismiss()
            }
            saving = false
        }
    }
}

private struct IntervalEditor: View {
    @State private var text: String
    let onApply: (Double) -> Void
    init(interval: Double, onApply: @escaping (Double) -> Void) {
        _text = State(initialValue: String(interval))
        self.onApply = onApply
    }
    private var value: Double? {
        guard let value = Double(text.replacingOccurrences(of: ",", with: ".")), value.isFinite, (0.25...60).contains(value) else { return nil }
        return value
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 13) {
            Text("刷新间隔").font(.system(size: 13, weight: .semibold))
            HStack {
                TextField("0.25–60", text: $text).textFieldStyle(.roundedBorder).frame(width: 110)
                Text("秒").font(.system(size: 12)).foregroundStyle(.secondary)
                Spacer()
                Button("应用") { if let value { onApply(value) } }
                    .buttonStyle(.borderedProminent).disabled(value == nil)
                    .keyboardShortcut(.defaultAction)
            }
            Text("0.25–60 秒").font(.system(size: 10)).foregroundStyle(.tertiary)
        }
        .padding(18).frame(width: 280)
        .tint(FleetPalette.accent)
    }
}
