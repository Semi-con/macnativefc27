# 从源码复现开发环境

需要自己有权使用的同版本游戏、密钥和固件。这里不会下载或分发它们。以下步骤生成的所有资源、日志、测试 App 和缓存都保存在仓库外；不能提交或作为 Release 附件上传。

测试环境为 Apple M1 Max、32 GB 内存、macOS 27.0.1。其他机器和系统版本需要单独验证。请预留数 GB 的本地构建/资源空间以及游戏输入本身所需空间。

## 构建和单独的离线解码环境

```sh
brew install rust python@3.14 cmake pkg-config opus
python3.14 tools/build.py
python3.14 tools/check_release.py
```

选定仓库外的新目录，单独安装离线解码器。该 Python 环境使用 GPL-2.0 的 `zbic`；它通过独立进程和磁盘文件与 GPL-3.0 的 Rust 工具交接，不打入运行时/App。

```sh
FC27_LOCAL="$HOME/Developer/fc27-local"
mkdir -p "$FC27_LOCAL"
python3.14 -m venv "$FC27_LOCAL/decoder"
"$FC27_LOCAL/decoder/bin/python" -m pip install -r tools/decoder-requirements.txt
```

## 提取、规范化、改写和资源实验

把以下示例路径替换成自己的输入。支持的身份见 `games/fc27/identity.json`。组合 XCI 可以同时作为 base、update，DLC 在启动时另传；分离文件分别填写。当前配方严格匹配这组已经分析过的资源，不适用于任意版本。

```sh
python3.14 tools/prepare_fc27.py \
  --base /path/to/own/base.xci \
  --update /path/to/own/update.nsp \
  --keys /path/to/own/prod.keys \
  --decoder-python "$FC27_LOCAL/decoder/bin/python" \
  --out "$FC27_LOCAL/stage" \
  --menu-experiment
```

`stage` 必须是新目录。流程包括：

1. `nxfs` 提取更新后的 ExeFS，核对 main build ID 与 NPDM SHA-256。
2. 独立规范化 main/rtld/sdk，对九个原始段逐一核对 NSO 头中存储的 SHA-256。
3. 从合并更新的 RomFS 提取 `Engine.Render.Core2.PlatformNvn.nro`，核对更新版 renderer 哈希。
4. `nxpatch` 改写四个模块，核对与开发记录相同的四个模块输出哈希。
5. 可选的菜单配方从本机原始 CAS/TOC 解码资产，修改四条事件路由、已有子关卡引用和一个自动准备布尔字段；更新相关索引与校验值。对输入和四个完整输出均核对 SHA-256，要求输出与开发实验 232 完全一致。

配方含格式操作、偏移、标识符和哈希，不含原始/修改后的 EBX、CAS、TOC 或资源包。遇到任何校验失败会退出，不会把别的版本当成兼容版本。失败目录保留以便本机诊断；修正输入后选择另一个新目录重试。

公开 `nxpatch verify` 没有包含私有验证环境的可选 Unicorn 组件，会明确失败。准备流程中的完整哈希比对不能代替指令等价验证。

## 隔离启动

```sh
python3.14 tools/run_fc27.py \
  --prepared "$FC27_LOCAL/stage" \
  --firmware /path/to/own/firmware/registered \
  --dlc /path/to/own/dlc-or-combined.xci \
  --out "$FC27_LOCAL/run-001" \
  --menu-experiment
```

DLC 可选，可重复 `--dlc`。默认生成全新 profile，使用全新的着色器缓存。需要在首次运行时自己处理协议、语言和初始设置；不自动同意协议。若要继续自己的已有进度，可显式增加 `--profile-source /path/to/own/profile`，脚本复制到本次 run，原目录不被当成可写存档。

程序在新的本地测试 App 中运行，使用源码里的版本/进程保护和独立测试控制通道，最长默认 1200 秒。不会覆盖 `/Applications` 下的游戏。窗口可正常退出；超时或互斥运行请求会结束测试进程。退出码为零只表示进程正常结束。

运行期间可用键盘/已配对的手柄。开发控制示例：

```sh
python3.14 tools/control_game.py "$FC27_LOCAL/run-001" --buttons A --hold-ms 250
python3.14 tools/control_game.py "$FC27_LOCAL/run-001" --frame --wait-frame
python3.14 tools/control_game.py "$FC27_LOCAL/run-001" --inspect-render-quality
```

截图等证据只留本机。冷缓存首次编译可能明显停顿，不能将首次加载窗口作为稳定帧率证据。

## 如何理解“复现到当前进度”

应分别核对：源码能构建；九个规范化段、四个改写模块和四个资源文件哈希相符；首次 3D 菜单确有连续动画；能够走过开球、球队/球衣选择并进入离线比赛；退出比赛后再次进入主菜单的行为如实记录。

`NX_FC27_RENDER_FPS=60` 只是原引擎目标与主机节奏配置，不保证达到 60 FPS。1920×1080 显示也不能证明比赛内部原生 1080p：还须使用渲染诊断核对动态分辨率关闭、比例为 1、实际主要渲染目标及 viewport 为 1920×1080。开发记录中的动态分辨率编辑为有时限的测试操作，不是已部署的永久设置。

当前未解决：原生 1080p 稳定 60 FPS、退出比赛后的第二轮菜单 Prep/Start、实体手柄和主观音画同步整体验收。移交这些源码不意味着这些问题已经解决。具体文件与下一步见 `HANDOFF.md`；公开导出版本的独立复现证据见 `REPRODUCIBILITY.json`。
