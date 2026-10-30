// Local app entry point. Python only supervises the native Rust executable.
import AppKit
import Foundation
import Darwin

// The game creates its own AppKit window in a child process. Activate that
// window once it finishes launching so a Finder launch accepts keyboard input.
if CommandLine.arguments.count == 3 && CommandLine.arguments[1] == "--activate-child",
   let pid = Int32(CommandLine.arguments[2]) {
    let deadline = Date().addingTimeInterval(10)
    while Date() < deadline {
        if let child = NSRunningApplication(processIdentifier: pid) {
            if child.isTerminated { exit(1) }
            if child.isFinishedLaunching {
                exit(child.activate(options: []) ? 0 : 1)
            }
        }
        Thread.sleep(forTimeInterval: 0.1)
    }
    exit(1)
}

func fail(_ message: String) -> Never {
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    let alert = NSAlert()
    alert.messageText = "FC 27 无法启动"
    alert.informativeText = message
    alert.addButton(withTitle: "好")
    app.activate(ignoringOtherApps: true)
    alert.runModal()
    exit(1)
}

guard let resources = Bundle.main.resourceURL else { fail("未找到应用资源。") }
let config = resources.appendingPathComponent("launch-config.json")
do {
    let object = try JSONSerialization.jsonObject(with: Data(contentsOf: config)) as? [String: Any]
    guard let python = object?["python"] as? String,
          FileManager.default.isExecutableFile(atPath: python) else {
        fail("未找到安装时使用的 Python 运行环境。请查看应用内的安装记录。")
    }
    let script = resources.appendingPathComponent("launch_game.py")
    let arguments = [python, "-B", script.path, config.path] + Array(CommandLine.arguments.dropFirst())
    var pointers = arguments.map { strdup($0) }
    pointers.append(nil)
    pointers.withUnsafeBufferPointer { buffer in
        python.withCString { path in _ = execv(path, buffer.baseAddress!) }
    }
    fail("无法启动本地运行环境：\(String(cString: strerror(errno)))")
} catch {
    fail("无法读取本地启动配置：\(error.localizedDescription)")
}
