# 材料清单与源码来源

## 仓库已包含的类别

- `sources/h20_sources_20260927.tar.gz`：2026-09-27 从原服务器收到的修改版 Megatron/Galvatron 源码快照，2,331 文件；完整 LICENSE 保留。使用方法和审查范围见 [SOURCE_RECEIPT_CN.md](SOURCE_RECEIPT_CN.md)。
- 中文交接、八模型参数、公共空间 CPU 枚举工具、标准库测试。
- `reference/common_exact/`：2026-09-23 修订后的公共空间评估/原生 Galvatron Profile/搜索适配逻辑。
- `reference/legacy_support/`：上述框架调用的旧 worker/env/launcher/probe 参考；含旧站点路径，**不可直接在 H20 执行**。
- `reference/qwen3_vocab_fix/`：Qwen3 padded vocab 151936 修复后的 launcher。
- `reference/cases32/`：此前三个受限用例的模型与 launcher 参考，不是 H20 32卡任务。
- `reference/project_custom_snapshot/`：本地持有的 test_parallel_model.py、dtsir_collect.py、Megatron initialize.py 部分快照，不是完整项目。
- `reference/space_ladder/`：内部空间递增先导逻辑，待八卡八模型参数化。
- `reference/runtime_checks/`：依赖审计、A100 kernel/native loop、通信测量实现参考；不照搬 ARM/module/账户参数。
- `source_manifest.json`：逐文件本地来源、原文件与交接文件的 SHA256、大小。为公开交接，历史个人账户和节点标识统一替换为 LEGACY 占位符；因此参考文件不是原件的逐字节副本，清单记录这一转换。

旧文档/注释描述的是旧实验，当前执行规则以本仓库 HANDOFF 和八模型配置为准。不要执行旧说明里的 sbatch、conda install 或覆盖命令。

## H20 主机仍需接收的私有输入

1. 三套 tokenizer 目录：`model_from_hf/llama2-hf`、`llama3-hf`、`qwen3-hf`。本地已核对16文件，单独私有传输。
2. 三组预分词数据：`dataset/{llama,llama3,qwen3}/enwiki_text_document.bin` 和 `.idx`。预期 hash 见 sources/input_manifest.json，本地数据包验证状态见 sources/review.json。

不需要完整预训练权重；这是随机初始化的性能训练。原数据约1.49 GiB，不应直接提交普通 Git。tokenizer/数据的再分发权限需用户确认；默认单独传到受控 H20 工作目录，GitHub 只留清单和 hash。

用户已在原服务器成功生成过两个包，可直接检查复用，无需再做实验：

```text
~/run/wjy/a100_migration_20260924_160432_p3rv8ih0/a100_sources.tar.gz
~/run/wjy/a100_migration_20260924_160432_p3rv8ih0/a100_training_data.tar.gz
```

旧 a100_sources 不再作为本次主源码，优先使用新 H20 快照。a100_training_data 可以在六个文件 hash 一致后复用；没有上传 GitHub。包名不能作为完整性证明。

## 原服务器重新收集（可选，不压缩、不训练）

将本仓库的 `scripts/collect_h20_sources.py` 和 `config/cases8.json` 保持目录关系复制到原服务器，运行：

```bash
SOURCE_ROOT="$HOME/run/wjy"
python scripts/collect_h20_sources.py \
  --megatron "$SOURCE_ROOT/Megatron-LM" \
  --galvatron "$SOURCE_ROOT/dependencies/Hetu-Galvatron-dtsir" \
  --out-parent "$SOURCE_ROOT" \
  --with-inputs --with-data
```

工具只读原项目，输出新的目录和 hash 清单。源码置于 `source_payload/`，tokenizer/数据置于 `private_inputs/`；不得把后者整目录提交 GitHub。疑似密钥、根目录外链接或缺少必要文件会明确报错，不静默发布不完整包。检查报告再把材料复制到 H20。当前 GitHub 仓库为公开：完整服务器源码需再次核对公开范围、敏感配置及第三方许可证后才提交，不因为扫描通过就默认允许公开一切；保留 LICENSE/NOTICE。

## 不迁移

旧 Profile 数值、conda 目录、wheel/.so、CUDA 编译缓存、checkpoint/权重、历史全量日志、代理/SSH/PAT/云密钥。没有完整源码不能靠复制 `.so` 弥补。
