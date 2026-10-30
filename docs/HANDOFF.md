# 开发移交

## 当前能力与问题

私有开发环境已进入离线比赛；菜单实验 232 在首次启动能播放原游戏 3D 人物动画，离开主页时通过原始 UnPrime 返回路径清理。退出比赛后可观察到第二轮 Prime，但未得到第二轮 Prep/Start，尚未修复。原生 1080p 比赛暖运行短窗口约 46.1 FPS，GPU 约 14.32 ms，未达到稳定 60 FPS。这些记录不等同于公开导出版本的完整验收。

最近两处同步修改：Prime 完成但前台取消时，不再等待可能永远不会请求的 Prep，改为调用原始 UnPrime，并继续要求真实清理返回；只读诊断增加底层 source 是否挂接、运行时 +0x40/+0x4c 和 +0xc9..+0xcc 原始字段。这些字段不被当作虚构完成事件。初次真实播放曾观察到 3/3、[0,0,1,0]；仍需采集第二轮状态验证因果。

## 代码位置

| 位置 | 用途 |
|---|---|
| `tools/prepare_fc27.py`、`normalize_nso.py` | 外部输入、独立解码、规范化和模块改写 |
| `tools/prepare_menu.py` | 精确资源集的菜单实验 232 配方 |
| `tools/run_fc27.py` | 当前实验开关、隔离 profile/缓存、签名测试 App、运行记录 |
| `crates/nxhost/src/fc27_menu_auto_lifecycle.rs` | 原始前台意图、Prime/Prep/Start/UnPrime 状态机 |
| `fc27_menu_prime_probe.rs`、`fc27_menu_lifecycle_observation.rs` | 原始方法与完成/清理路径保护 |
| `fc27_menu_shadow_bindings.rs` | 精确空 shadow 子视图从真实父视图继承缺失描述符 |
| `fc27_menu_ambassador_source.rs` | 使用原有选中球员记录填补空结果 |
| `menu_cinematic_diagnostics.rs` | 有边界的只读类型/运行状态诊断 |
| `crates/nxhost/src/nvn/`、`crates/nxshader/` | Metal 资源、调度与着色器转换 |
| `tools/control_game.py` | 受保护的本地输入与有时限的诊断操作 |

除完整前缀所示外，菜单 Rust 文件位于 `crates/nxhost/src/`。实验代码默认关闭；`run_fc27.py --menu-experiment` 接齐前置条件。不要移除精确版本、原始方法和真实完成事件保护来绕过失败。

## 下一步验证

1. 对第二轮 Prime 后缺失 Prep 的场景采集原始事件与新增原始字段；分别核实原始前端的 prepare 请求、选中记录和底层 source 状态。
2. 单独测量比赛 CPU 热点，采样窗口避开整堆渲染诊断、截图与首次管线编译。此前重叠采集样本不能用于热点归因。
3. 在实际原生 1080p 比赛场景核实渲染设置、目标与 viewport，再取暖运行帧率；不能只凭显示窗口或 60 帧配置宣称达标。

私有运行记录、截图、原始资源、着色器捕获、游戏模块、密钥、固件和存档均未公开。读取本人的输入并重新产生这些本地产物是复现的一部分。公开测试中移除的可选验证器/捕获夹具见 `PROVENANCE.json`；不要将未执行或忽略的测试计为通过。
