# H20 八模型最终对比：30 GiB/s 参数版

本报告为唯一当前结果版本：8 个模型 × 公共/完整空间，共 16 项 Devastator（Megatron 后端）与官方 Galvatron 对比。

经验成本参数 `BAND_WIDTH_MEMORY_TRANS=30*1024**3` 字节/秒，即 30 GiB/s（32.21225472 GB/s），不是实测 HBM 带宽。固定候选与已有算子 profile 完成全部候选重新评分；选中配置匹配已有训练证据后复用，不声称新增了 16 次独立训练。

建模灵感来源：NVIDIA 在 [CUDA C++ Best Practices Guide 的有效带宽定义](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#effective-bandwidth-calculation)中，以对应活动的读写字节量除以耗时计算有效带宽，即 `B_eff = (B_read + B_write) / t`（换算为 GB/s 时再除以 10⁹）。这一“数据量与时间通过等效速率关联”的思路，是本模型原始 20 GiB/s 参数（代码为 `20*1024**3` 字节/秒）的建模灵感来源。将其用于建模算子间的数据相关等待，是本工作的经验建模选择；NVIDIA 资料并未给出 20 GiB/s 这一取值，也未将该公式定义为训练算子间等待的统一速率。本报告仍采用上文的 30 GiB/s 参数版本。

| 模型 | 公共 Devastator / Galvatron（秒/步） | 完整 Devastator / Galvatron（秒/步） |
|---|---:|---:|
| llama7b_2k | 25.679 / 26.517 | 22.064 / 25.050 |
| llama2_7b_4k | 54.291 / 55.724 | 47.103 / 53.062 |
| llama2_13b_4k | 201.211 / 204.298 | 177.141 / 205.023 |
| llama3_8b_8k | 33.655 / 33.655 | 32.791 / 32.887 |
| qwen2_1p5b_32k | 56.462 / 56.462 | 55.767 / 57.342 |
| qwen2_7b_32k | 169.360 / 169.360 | 168.102 / 173.826 |
| qwen25_14b_8k | 61.410 / 62.335 | 60.085 / 67.591 |
| qwen3_14b_4k | 54.140 / 58.161 | 56.153 / 55.103 |

公共空间：5 项实测更快，3 项按同配置复用 Galvatron 值相等；完整空间：7 项更快，Qwen3-14B-4k 更慢。完整空间有 7 项优于自身公共空间，Qwen3-14B-4k 为例外。小幅差异不作统计显著性结论。

## 读取与绘图

- [comparison.csv](comparison.csv)：16 项配对成绩、配置、复用标记；[results.csv](results.csv)：双方 32 行成绩与内存峰值。
- [rank_steps.csv](rank_steps.csv)：逐 rank、逐 iteration 的耗时、loss、显存等；[configurations/](configurations/)：本版 16 项预测器获选配置。
- [evidence/](evidence/)：按模型打包的原始日志、实际参数、更新证据、候选评分、profile 和 Galvatron 层级策略，SHA256 清单可校验。

## 测量与复用约定

每个实际训练证据为一次独立训练、10 步，统计第 6–10 步。主表采用双方一致的 CUDA 同步整步耗时，每步取 8 rank 最大值后求平均；不混用不同计时范围。Megatron 原生 `performance(samples/s)` 另存 `native_samples_per_second`，未以同步耗时倒数冒充原生输出；复用 Galvatron 的三项该字段留空。

公共空间采用双方对齐的原生 KV heads、FlashAttention 与小 mask 后端约定。完整空间开放各自支持的优化；Devastator 可搜索 CP、UP、优化器分片、重计算和 GQA，KV 下界 min(8, 原生 KV heads)。结构变化与两侧分片粒度差异保留在配置中，不将不同结构或分层分片写成相同配置。

原始训练复用依据为同模型的完整并行向量和策略一致；三项公共空间跨框架复用按用户批准的同配置等值约定，明确标记 `equal_by_reuse`，不是两次独立测量。所有原始证据保留原路径与历史预测字段，仅 configurations/ 和 ranking.json 表示本次选中结果。

本次为离线重评分及已有训练证据复用，没有新的端到端搜索墙钟测量，Devastator 的本版 search_e2e_seconds 留空；不以历史搜索时间冒充本版时间。通信标定不计入双方搜索耗时。

## 复现

先运行 `python replay/verify.py` 校验发布文件、证据归档和 16 项数据。解压对应 evidence/*.tar.gz 到新目录，然后用 `replay/replay_prediction.py` 对归档候选与 profile 重放本版预测（需要 PyTorch、NumPy、pandas、SciPy、scikit-learn；不启动 GPU 训练）。

训练重放的原始启动命令在 measurements/*/*/*/stages/，实际生效参数在 rank*.json；须将原服务器路径映射到本地运行环境和私有输入，并使用新的输出目录。入口与环境依赖在 replay/。私有训练数据、tokenizer、权重和凭据不在本发布中。

旧版结果保留在 Git 历史中，不再作为当前对比入口。
