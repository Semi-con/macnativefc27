# Source export validation — 2026-10-30

本记录仅对应公开导出源码的工具构建与测试。编译结果、缓存和原始日志保存在仓库外，不作为公开附件提供。

已在 Apple M1 Max / macOS 27.0.1 执行：

```sh
cargo build --workspace --release --locked
cargo test --workspace --release --locked -- --test-threads=1
swiftc -O -framework AppKit launcher/FC27Launcher.swift -o <private-output>
python3 -m py_compile tools/build.py tools/check_release.py tools/control_game.py launcher/launch_game.py launcher/game_slot.py
python3 tools/check_release.py
```

- Rust 工作区 release 构建成功。
- 单元/文档测试汇总：237 passed、0 failed、67 ignored。
- 被忽略的测试包括需实际 Metal GPU、专用运行开关或外部捕获输入的测试；没有把这些测试记为通过。
- Swift 启动器源码编译成功；列出的 Python 源码语法检查成功。
- `nxfs`、`nxpatch`、`nxhost` 的帮助入口正常；CAS 工具缺少参数时明确报错。
- `nxpatch verify` 明确拒绝执行缺省的可选验证器，没有生成等价性通过结论。
- 发布清单与文件内容核对通过；依赖元数据中没有 Unicorn、未声明许可包或 GPL-2.0-only 组合。

这不是公开导出版本的真实游戏验收。没有使用这份导出程序重做完整比赛、画质、音频、实体手柄和读档验收，也没有验证原生 1080p 稳定 60 FPS。私有开发版本的结果不能替代这些缺失证据。
