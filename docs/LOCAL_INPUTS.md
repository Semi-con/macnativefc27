# 本地输入与开发运行

这是开发入口说明，不是已经完成验收的一键安装程序。

把有合法使用权的游戏输入、密钥、固件、规范化模块、改写输出和用户状态保存在仓库外。这里只传密钥文件路径给本地程序，不填写或输出密钥内容。

目标 NSO build ID 和 NPDM 哈希会受到严格检查；文件名或版本标签相同并不代表输入兼容。ZBIC 编码的 NSO 需要先在本地规范化并验证各段哈希，当前 `nxfs` 遇到未规范化输入会明确拒绝。解码器及输出不随仓库分发。

工具构建后，可对本地已规范化模块调用普通重写入口：

```sh
target/release/nxpatch rewrite /path/to/normalized/main /path/to/normalized/rtld /path/to/normalized/sdk /path/to/normalized/renderer --out /path/to/private/prepared
```

以下运行命令只展示参数接线；需要使用者已经准备并验证的本地输入，不能据此推断任意游戏版本都能运行：

```sh
target/release/nxhost \
  --patched /path/to/private/prepared \
  --npdm /path/to/private/exefs/main.npdm \
  --keys /path/to/private/prod.keys \
  --base /path/to/private/base.xci \
  --update /path/to/private/update.nsp \
  --save /path/to/private/profile/save \
  --sdmc /path/to/private/profile/sdmc \
  --system /path/to/private/system \
  --firmware /path/to/private/firmware
```

DLC 可通过重复的 `--dlc` 参数提供本地路径。修改后的资源和主菜单实验资产必须在本地从原输入生成，本次公开导出不包含这条私有资源准备流水线或生成结果，因此不能直接复现私有开发程序的全部菜单实验。

主菜单生命周期、渲染字段编辑等实验有测试进程/精确版本保护；普通构建不会因此自动启用全部实验。请阅读对应 Rust 模块中的开关与前置条件，不要删除保护来伪造验证结果。
