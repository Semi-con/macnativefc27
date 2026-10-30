# macnativefc27

EA SPORTS FC 27 在 Apple Silicon macOS 上的实验性原生兼容层与开发工具。
基于 [Mush8](https://github.com/c22dev/mush8) 的 Rust/Cargo 架构修改，包含 AArch64 离线改写、Horizon 服务、NVN → Metal、着色器转换、输入、音频和版本诊断。

Experimental FC 27 compatibility work for Apple Silicon macOS. Source and tools only; bring your own legally usable game data. This is a development snapshot, not a ready-to-play game distribution.

## 内容与许可

- 本仓库提供兼容层和工具源码、版本身份/哈希信息，不提供游戏镜像、游戏程序模块、资源、着色器捕获/翻译结果、密钥、固件或存档。
- 提取、指令改写、资源处理和游戏着色器转换的结果只能由使用者在自己的电脑上从有权使用的输入生成；不要上传这些产物。
- 继承的 Mush8 源码与本项目修改按 GPL-3.0 发布，保留上游版权说明。各第三方组件保留自己的许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- 参考 MacNativeGames 的发布内容边界；本仓库是独立项目，不是其 MIT 仓库的组成部分，也不宣称满足该仓库拒收 GPL 代码的贡献规则。
- 未获 Nintendo、EA、Apple 或 MacNativeGames 授权、认可或赞助。产品名称仅用于说明兼容目标；商标属于各自权利人。
- 公开源码不等于已获得游戏、固件或技术措施相关授权。使用和发布须遵守适用法律及许可；本项目不提供游戏或密钥下载渠道。

## 当前状态

已在 Apple M1 Max / macOS 27.0.1 的私有开发环境进入离线比赛，验证了部分控制输入、菜单及音频输出路径。目标输入身份见 [games/fc27/identity.json](games/fc27/identity.json)。

尚未完成的项目包括：

- 原生 1920×1080 稳定 60 FPS。最近一次原生比赛窗口约 46 FPS，不能宣称已达到 60 FPS。
- 连续 3D 主菜单实验在首次启动可播放；退出比赛后重新准备/播放的生命周期仍未完整验证通过。
- 实体手柄、主观音效/解说同步与最终新进程读档的整体验收。
- 面向普通使用者的最终安装与完整游戏验收。开发者资源准备和隔离启动流程见下文；这与完成上述游戏目标是不同的验证项目。

公开导出移除了可选的 Unicorn 等价验证组件和来源待进一步确认的捕获相关 GPU 测试夹具；开发主工程保持独立。详细差异见 [PROVENANCE.json](PROVENANCE.json)。原开发程序的运行记录不能证明这一导出版本已经通过游戏验收。

## 构建源码工具

需要 Apple Silicon Mac、Rust/Cargo、Xcode 命令行工具，以及 Homebrew 的 `cmake`、`pkg-config` 和 `opus`。只构建源码工具不需要游戏或密钥。

```sh
brew install cmake pkg-config opus
python3 tools/build.py
python3 tools/check_release.py
```

如 libopus 不在 Homebrew 默认位置，可设置 `NX_OPUS_LIB_DIR` 指向含 `libopus.a` 的目录。

运行时与工具输出在本机 `target/release/`。命令入口可分别查看：

```sh
target/release/nxfs --help
target/release/nxpatch --help
target/release/nxhost --help
```

普通重写入口保留；`nxpatch verify` 会明确报错说明公开导出未包含其可选等价验证器，不会伪报验证成功。

## 本地输入与运行

见 [完整复现步骤](docs/REPRODUCE.md) 和 [开发移交](docs/HANDOFF.md)。提供提取、独立离线规范化、四模块改写、精确菜单资源配方和隔离测试启动入口；使用者从自己的输入在本机生成全部游戏派生产物。源码不寻找开发者电脑的路径，不自动下载游戏、密钥或固件。

`tools/run_fc27.py` 在本机生成签名测试 App 和独立 profile/缓存，并接齐当前实验所需的保护开关。`launcher/` 保留普通启动器源码与互斥运行辅助模块。仓库不提供游戏 App 或开发者配置。当前公开构建说明只覆盖 macOS。公开导出版本的验证范围见 [复现记录](docs/REPRODUCIBILITY.json)。

## 参与开发

请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [PUBLICATION.md](PUBLICATION.md)。不要在 issue、提交、附件或 CI 里上传游戏数据、密钥、存档和未经审查的游戏派生文件。
