# 八卡 H20 公共 / 完整空间对比

本轮目标为八模型 × 两系统 × 两空间的 32 行状态与证据，不把预测值当作实测，不把小模型冒烟测试当作正式结果。

工作目录：`/opt/hbv/trainir-h20-experiments/20260927-v1`。八例及原生 KV、GBS、序列、输入见该目录 `cases8.json`。`protocol.json` 记录实际运行协议；初始规划版本单独保留为 `protocol.initial.json`。双方相同显存筛选预算 85 GiB，训练 1×10 步，实测统计 6–10 步。搜索耗时按用户确认口径，采用从流程进程启动到获选配置可启动训练的端到端墙钟时间，包含所有配置准备、脚本生成、初始化、计算/显存 Profile、处理、进程启动、搜索和结果导出；提前完成的通信校准及后续训练不计入。`search_e2e_seconds` 是主比较指标，分阶段合计仅作诊断。早期运行由原始日志创建时间恢复边界并标记，后续按内核进程启动时钟直接计时。计算/显存 Profile 使用各自原生策略；Devastator 原生算子 50 次预热和 50 次测量，Galvatron 原生计算 Profile 20 步及其原生过滤，显存 Profile 迭代索引 5 导出。初始规划中的 5+5 Profile 未执行，已在首个正式成功训练之前更正协议，未改写测量数据。

## 搜索空间

公共空间使用 `common_space.py` 精确核对双方实际候选：TP/PP/MBS ∈ {1,2,4,8}，DP×TP×PP=8，SP 跟随 TP，CP/UP/EP=1，原生 KV 固定、无重计算/优化器分片/VPP；均匀层划分。该版 Galvatron Attention 即使 TP=1 也要求 sequence_parallel=True；运行时保留此开关，TP=1 时 SP 度数仍为 1、切分为恒等操作，不能因此删除 TP=1 候选。八例共 296 个结构候选，OOM/预测不可行仍须留痕。

Devastator 完整空间包含公共空间，并开放合法 CP/UP、GQA（KV 下界 min(8,原生 KV)，上界 Q heads，取合法约数）、选择性 MLP 重计算、distributed optimizer、合法 VPP；仍保留上述四个 MBS 值。GQA 改变结构，记录获选实际 KV，不能把该收益全部解释为同模型并行优化收益。未实现 offload 不计入空间。

Galvatron 完整空间使用原生逐层策略搜索、checkpoint、DDP/Zero3、独立 SP/Ulysses、embedding/lmhead 策略及原生 chunk 枚举；原生 KV 固定，TP 和 SP 互斥，流水线使用原生均匀划分和 pipedream_flush。SP 必须整除 Q heads；其运行时可复制 KV 来支持 SP 大于原生 KV。CP 在本版 Galvatron 原生策略导出中丢失，往返核对会退化为 CP=1，因此完整空间固定 CP=1，不声称 CP 可搜索；证据见根目录 `galvatron_full_capability_audit.json`。原生搜索输出 `embed_sdp` 而运行时读取 `vocab_sdp`，适配器将选中值显式传入已有运行时配置字段。具体有效列表以每例 `space_definition.json` 为准；需要完整空间小样本验证后扩大运行，不能只解除开关便宣告支持。

## 留存和重放

- `model_audit/`：两后端实际解析模型配置、逐项核对。
- `input_receipt.json`、`input_audit.json`：私有输入的哈希与词表/数据检查，输入内容不入 git。
- `hardware/`：Galvatron 原生校准、NCCL-tests 源码版本/构建记录、原始日志与 Devastator 原生拟合结果。hypercube 单线程八 GPU 正确性失败；同一二进制改为八线程各一 GPU 通过，失败日志保留。该项原日志 redop 为空，仅在拟合输入补 `none` 对齐列，不修改数值或公式。
- `runs/<case>/<system>/<space>/<attempt>/`：独立冷缓存、完整命令/环境、候选搜索、获选配置、训练逐卡逐步证据。
- `stages/*.gpu.csv`：每秒采样八卡显存、利用率、温度、功耗；`*.json` 记录退出码和耗时；新增 stage 同时保存 `*.code/` 脚本快照。
- `training_retries/`：修复观测器后单独重放获选配置，复用原搜索、不删除失败。训练证据采样在计时区间外，每张量最多 64 个整数索引点；大张量不能用 FP32 linspace 生成边界索引。
- `results.json/csv`：32 行当前状态；`steps.csv`：逐步实测；`rank_steps.csv`：逐 rank 的 loss/梯度/时间记录；`candidates.csv` 和 `profile_accounting.csv`：候选预测与 Profile 开销，由 `h20_export_search_tables.py` 生成；`attempts.json/csv`：全部尝试。由 `python scripts/h20_summarize_campaign.py` 更新。
- `paired_results.json/csv`：八模型 × 两空间的 16 行成对比较，随汇总自动更新；双方训练完成才计算耗时比，缺失测量保持空值。保留实际 KV、计时来源、显存指标和训练证据路径；完整空间改变 KV 时不能据此推断模型质量等价。

入口：`h20_run_megatron.py --case ID --space common|full --attempt NEW`；Galvatron 使用 `/opt/hbv/venv-galvatron-h20/bin/python scripts/h20_run_galvatron.py`，并将 `PYTHONPATH` 指向接收的 Galvatron runtime。训练重试入口 `h20_retry_training.py --run PATH --retry NEW`。每个入口拒绝覆盖既有尝试。八卡作业串行执行，防止测量互扰。

当前完成情况以 `results.csv` 和 `completion_audit.json` 为准，不能根据状态表中的 pending/running 行宣称 32 组完成。审计同时检查端到端搜索时间覆盖串行 Profile 和搜索阶段，历史恢复时间必须明确标注。先公共空间，再完整空间。失败/超时要记录原因，修复明确的集成问题再重试；资源不可行保留真实状态。

长序列显存 Profile 补充：Qwen2 1.5B/7B 的原始全序列原生显存 Profile 发生 CUDA OOM，原尝试和计时保留。后续这两例及 Qwen2.5 14B 使用 Galvatron 原生 sequence 模式，在 1024/2048/4096 序列分别测量，再由未修改的原生处理器及搜索器外推至目标序列。该模式原生只实测 TP1，并推导高 TP 数据；这是成本模型近似，不能替代最终八卡训练验证。计算 Profile、正式搜索目标和训练仍用原模型及目标序列。具体策略见 campaign 的 galvatron_profile_policy.json，每次运行保存 profiler args；全部 Profile 和处理耗时计入搜索端到端时间。

重试计时：results/paired_results 同时导出 search_cumulative_e2e_seconds（该模型、系统、空间所有正式搜索尝试的端到端时间之和，包含失败尝试），避免重试后只展示成功一次的费用。search_e2e_seconds 保留当前单次尝试口径。若仍有进行中或缺失计时的尝试，累计完整值为空，另列已记录时间及完整性标记；训练重试和独立调试重放不混入搜索计时，原日志仍保留。

公共空间阶段验收见 common_phase_audit.json：逐模型核对双方实际候选集合完全一致，并关联成功训练审计或原生无候选结果审计。含已通过审计的训练重试。该阶段验收不会把无候选结果当作训练成功，也不代表完整空间完成。每次汇总自动刷新。

完整空间真实数据 RoPE 适配：首项 Ulysses4/KV2 验证发现 dataloader 把整段位置编码传给分片序列，产生长度 32/128 不匹配。适配器对获选 use_sp 布局传 rotary_embedding=None，调用未修改的上游逐层 _get_rotary_pos_emb，使用该层实际 TP/SP 组和位置偏移；搜索、原生算子和公共空间保持原路径。失败证据见 preflight/galvatron-full-training-01；新路径须在独立的 -02 验证目录通过八卡验证后才开展 Galvatron 完整空间正式运行。

上述 RoPE 适配现已通过独立八卡验证：Ulysses4/KV2、Zero3+checkpoint、异构逐层 TP/DP、Zero3 词表层分片四例各十步，检查非零参数更新与分片状态，证据见 galvatron_full_validation_audit.json。当前完整空间队列状态以 full_queue_02.json 为准，批次 full-02 复用 full-01 已有尝试。功能验证不计为正式完整模型结果。

逐层绘图表 `galvatron_selected_layers.csv/json` 展开最新获选配置的 TP、独立 Ulysses、DP、checkpoint、Zero3 和词表分片，附配置哈希及当前结果状态。层号与流水线阶段号从零开始；Ulysses 与 TP 内部的 sequence_parallel 开关不同。`implied_micro_batch_size` 由 GBS/(该层 DP×chunks) 推导。这张表记录配置，不替代训练或实际分片证据。

获选配置执行审计：`completion_audit.json` 对 Devastator 八个 rank 的有效参数逐项核对获选 DP/PP/TP、CP×UP、CP/UP 通信类型及层级、MBS、KV heads、重计算、优化器分片与 VPP；同时核对八卡乘积及 GBS=DP×MBS×chunks。正式完整空间已出现 CP 与 VPP 获选，不能仅凭搜索空间声明认定训练采用这些设置。

统一重放入口索引 `replay_index.json` 覆盖 32 行及其全部已留存尝试、训练重试，集中列出原始命令、环境、cwd、阶段源码快照、日志及配置哈希。它是证据索引，不自动执行：重放应使用新的 attempt/retry 目录，不能原地执行历史命令覆盖结果。使用当前代码重放与恢复历史源码快照不同，须保留这一差异及原依赖环境。

OOM 统计按用户最新确认，以获选配置正式训练发生的实际 CUDA OOM 为主；搜索前置计算/显存 profiling 失败单列，搜索器预测无可行候选另列，不混入实测 OOM 数。`oom_incidents.json/csv` 保留历史事件，按工作流失败计一次，不按 rank 或诊断重放重复计数；`oom_reporting_policy.json` 保存口径。`results.csv` 中 `oom_*_gib` 来自原始异常中的舍入 GiB 数值，保留请求分配量、设备空闲、PyTorch 已分配和已保留但未分配缓存，不能当作精确字节值。

分配器对照入口 `h20_allocator_diagnostics.py`，输出 `diagnostics/allocator-expandable-01/`。等待正式八卡队列结束后，按原始源码快照和相同工作负载，只设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，输出目录单独隔离。先验证 Devastator/Megatron Qwen3 完整空间获选配置十步，再复核 Galvatron Qwen2-7B 原始显存 profiling。成功后将该 OOM 标记为内存管理敏感问题，不据此认定为固有容量预测错误，也不声称证明整个预测模型准确；原始失败和诊断结果同时保留。诊断结果不静默替换主表正式运行的配置、状态或性能。若需展示修复后的性能，必须注明分配器设置与独立运行证据。

`full_phase_audit.json` 检查八个完整空间成对结果，分别引用成功训练或已核实终止失败的底层审计；它不等于 16 项训练全部成功，也不表示双方完整搜索空间相同。

`allocator_diagnostic_audit.json` 核对分配器实验的原始源码快照、归一化命令一致性、分配器环境、原失败与新日志哈希。获选训练额外核对八个 rank/GPU、原始模型清单、有效训练参数和并行策略、十步有限数值、非零参数更新与 FP32 优化器参数，并重新计算后五步均值。通过验收的诊断性能单列 `allocator_diagnostic_metrics.csv/json`，不混入默认分配器主表。`REPORT_CN.md` 从当前证据表生成，运行中仍显示进行中状态，不能作为完成声明。

历史源码留存边界：前 30 个阶段未启用逐阶段源码快照，保留原命令、环境、配置、日志及 contract 中流程启动时脚本哈希。`source_recovery_index.json` 按 SHA256 关联现存快照中相同内容，并列出缺失项；它不会伪造历史文件，流程启动哈希也不能证明随后发生修改的阶段使用同一版本。其余 430 个正式阶段已有源码快照。需要精确历史源码的重放必须检查这一边界，使用现行集成脚本重放须明确标注。
