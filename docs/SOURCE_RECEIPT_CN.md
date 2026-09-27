# 2026-09-27 源码接收与使用

## 已收到及已核对

- 原服务器导出源码 2,331 文件，逐文件 SHA256 与采集清单一致。
- Megatron 基准 commit `1f6cde85d23ff0c307a47bbdd8bfd778b95a161f`，Galvatron 基准 commit `cea12ffb146a220643c8f99f9cb84294755d29f8`；两者均包含工作区修改，必须使用快照。
- 两项目 LICENSE 和源码版权说明保留。
- 两处 `schema_hf.py` 告警是视觉模型 `class_token` 的 checkpoint 参数路径映射，不是凭证；按文件哈希和具体行人工确认，没有全局忽略 token 字段。
- `run.sh`、`test1.sh` 共四处 URL userinfo 已替换为环境变量占位符，服务器原文件未修改。它们不是 H20 入口，不要直接运行。
- 16 个 tokenizer 文件及旧数据包的六个训练数据文件均已逐文件校验通过，但不上传公开仓库。预期 hash 见 `sources/input_manifest.json`；最终数据包 SHA256 见 `sources/review.json`。数据验证晚于源码压缩，包内历史报告的 pending 状态以仓库外置 review.json 的最终 PASS 为准。

## H20 上使用

在交接仓库根目录执行：

```bash
python scripts/unpack_h20_sources.py --out work/runtime
```

输出目录必须不存在。脚本校验压缩包 SHA256，拒绝路径穿越和链接，再核对全部源码文件。得到：

```text
work/runtime/Megatron-LM/
work/runtime/Hetu-Galvatron-dtsir/
work/runtime/H20_SOURCE_MANIFEST.json
work/runtime/H20_SOURCE_REVIEW.json
```

这一步不安装环境，不启动 GPU，也不执行旧脚本。之后按 HANDOFF 完成环境和双系统八卡训练检查。

用户单独上传的 `h20_tokenizers_PRIVATE.tar.gz` 和 `a100_training_data.tar.gz` 可解到同一个 `work/runtime/` 下；两包使用 `Megatron-LM/model_from_hf/` 与 `Megatron-LM/dataset/` 相对路径。先确认数据包传输完成并校验，再解包。禁止把这两包或原始 `h20_LOCAL_REVIEW_ONLY.tar.gz` 提交 GitHub。

## 边界

快照含主要源码、自定义测试及搜索脚本，包括 dtsir_space_tests、dtsir_common16/32、paired16/32、dtsir_galvatron6。它不包含 conda、编译扩展、旧 Profile、训练结果或完整 Git 历史，也不保证任何未纳入采集器的任意未跟踪资源存在；实际依赖仍需接手 agent 核对。Galvatron 的 dtsir_galvatron32 不在收到的快照中，但八模型参数和三个较大用例参考已在交接包，单节点 H20 不需要原 32 卡提交入口。

源码上传不代表已完成 H20 适配、模型等价审计、训练验证或实验。公共空间和完整空间仍按同一八例推进。
