# TrainIR / Devastator：H20 八卡实验交接

目标：在**同一台单节点 8×H20 96GB** 上，先验证 Devastator/Megatron 与 Galvatron 的八卡训练，再以固定的八个完整模型运行公共空间和完整空间实验。

**当前唯一结果版本：** [八模型公共/完整空间 16 项对比（30 GiB/s 参数版）](reports/h20-final-20260929/README.md)。结果、配置、绘图 CSV 与原始证据均在该入口；旧版成绩不再作为当前结论，历史文件保留在 Git 历史。本轮使用官方 Galvatron v2.4.1（`cea12ffb146a220643c8f99f9cb84294755d29f8`）。

**现已包含原服务器修改版 Megatron/Galvatron 的源码快照（2,331 文件），但不是已经在 H20 验收的一键训练包。** 见 [源码接收与使用](docs/SOURCE_RECEIPT_CN.md)。tokenizer 和训练数据通过私有渠道传输，不在公开仓库。不要直接运行 `reference/` 或源码里的旧 Slurm/训练脚本。

## 接手顺序

1. 阅读 [交接文档](docs/HANDOFF_CN.md) 和 [H20 agent 首条任务](docs/AGENT_START_CN.md)。
2. 核对 [八模型清单](config/cases8.json)，用 `python scripts/unpack_h20_sources.py --out work/runtime` 解包经校验的源码；按 [材料清单](docs/MATERIALS_CN.md) 补齐私有输入。本仓库原有 `source_manifest.json` 仍只对应 reference 文件，源码快照使用包内独立清单。
3. 在 CPU 上执行 `python -m unittest discover -s tests -v`；生成八卡公共候选清单 `python scripts/common_space.py --out work/common8_manifest`（输出目录必须不存在）。
4. 在 H20 适配环境和启动器；先训练验证，再公共空间，最后完整空间。实际完成条件见交接文档，不以退出码 0 替代训练证据。

## 固定八例：两套实验全部保留

| 模型 | ID | 序列长度 | 全局 batch | 精度 |
|---|---|---:|---:|---|
| LLaMA-7B-2k | `llama7b_2k` | 2048 | 256 | FP16 |
| LLaMA2-7B-4k | `llama2_7b_4k` | 4096 | 256 | FP16 |
| LLaMA2-13B-4k | `llama2_13b_4k` | 4096 | 512 | FP16 |
| LLaMA3-8B-8k | `llama3_8b_8k` | 8192 | 64 | BF16 |
| Qwen2-1.5B-32k | `qwen2_1p5b_32k` | 32768 | 64 | BF16 |
| Qwen2-7B-32k | `qwen2_7b_32k` | 32768 | 64 | BF16 |
| Qwen2.5-14B-8k | `qwen25_14b_8k` | 8192 | 64 | BF16 |
| Qwen3-14B-4k | `qwen3_14b_4k` | 4096 | 128 | BF16 |

这是既定性能 benchmark 的模型定义；不是从 Hugging Face 名称重新下载并推断配置。尤其三个 Qwen 系列沿用历史 qwen3 tokenizer/data 映射，必须保留并核对有效参数。

## 不变的测量约定

- 每个选中配置 **1 次独立训练、10 个 iteration、统计第 6–10 个**；不擅自变成 3 次。
- 通信硬件预先测量，双方均不计入搜索耗时；计算/内存 Profile、处理和搜索实测墙钟计入，细项分别保存。
- H20 重新采集性能证据；不复用 5090/A100 Profile 数值，也不重跑或改写旧结果。
- 公共空间候选相同；完整空间保留各系统支持的优化，明确报告空间差异，不能称完全相同空间。
- 公共空间固定原生 GQA/KV heads。按 2026-09-27 用户修订，Devastator 完整空间开放 CP、UP 和 GQA，KV heads 下界为 `min(8, 脚本原生 KV heads)`；记录结构变化，不把结构搜索收益称为同模型吞吐提升。见 [修订与验证](docs/H20_CP_UP_UPDATE_CN.md)。
- 失败也保留在八例表中，区分配置/依赖错误、Profile 失败、无可行候选、选中配置 OOM 和训练成功。

## 仓库与凭证

用户选择公开仓库 [`gslkj3/trainir-h20-handoff`](https://github.com/gslkj3/trainir-h20-handoff)。任何人都能下载此交接包，无需令牌；它不包含训练输入或凭据。

推送修改仍需要 GitHub 身份认证。用户已允许省略令牌生成，本次不创建密钥；如之后需要独立的 30 天写入凭证，应只授权此仓库。**不在 Git、日志或聊天里保存令牌。** 方法见 [访问说明](docs/GITHUB_ACCESS_CN.md)。

源码引用保留原作者版权和许可证。本交接不为第三方软件或数据重新授权；没有上传完整权重、训练数据或 tokenizer。
