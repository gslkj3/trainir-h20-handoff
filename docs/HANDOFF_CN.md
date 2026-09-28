# H20 八卡实验交接文档

日期：2026-09-27。接手对象：在 H20 服务器上工作的 agent。

## 1. 要完成什么

按顺序完成以下三个阶段，使用 README 和 `config/cases8.json` 中的同一份八模型清单：

1. **两系统八卡训练验证**：确认真实模型前向、反向、optimizer update、TP/SP 和 PP 能工作。
2. **公共空间实验**：Devastator/Megatron 和 Galvatron 搜索完全相同的结构合法候选集合，各自选优后训练。
3. **完整空间实验**：两系统分别使用其实现支持的完整搜索空间；Devastator 按用户修订包含 GQA 结构搜索，记录搜索成本和选中配置质量；另以 Devastator 公共/完整空间构成内部空间扩大对照。

“完整”指原始层数的模型、完整搜索和获选配置短程性能训练，不是训练至收敛，不是继续增加到 10 万候选，也不是保证八例全部通过。该平台提供单节点 NVLink 证据，不代表跨节点扩展性。

## 2. 已知事实与缺口

### 已知

- 用户提供的机器输出为 8 张 H20，每卡 `97871 MiB`，GPU 对之间标记 `NV18`。实际运行时仍记录 GPU UUID、显存和拓扑，不能假设租用机器永远不变。
- 旧 5090 环境：Python 3.12、PyTorch 2.9.0+cu128、FlashAttention 2.8.1，Galvatron 2.4.1，Galvatron 基准 commit `cea12ffb146a220643c8f99f9cb84294755d29f8`。这些是可参考的旧环境，不是 H20 已验证环境。
- A100 使用过 aarch64/Python 3.10/Torch 2.5.1+cu121；该环境和 `.so` 不能直接复制到可能为 x86_64 的 H20 主机。
- 已收到原服务器修改版源码快照，共 2,331 个文件，保存在 `sources/h20_sources_20260927.tar.gz`。SHA256 及审查范围见 `sources/review.json`；`reference/` 仍是历史参考，不覆盖更新的源码。

### 尚未完成

- tokenizer 已在本地收到但不公开；数据包状态见 `sources/review.json`。H20 主机仍需接收并核对六个数据文件和三套 tokenizer。源码是带未提交修改的工作区快照，不是仅按上游 commit 可复原的版本；不能用上游源码替代。
- 未检测 H20 CPU 架构、容器/conda、CUDA toolkit、实际加载 NCCL、torch/TE/FA/Apex 兼容性。
- 旧公共空间脚本固定 16 卡、2×8、28 GiB 预算；旧完整空间 ladder 固定 4 卡、八层、SP off、3 次训练。**它们都不是 H20 入口。**
- 三个历史受限用例未经 common5 同等级的模型等价审计，Qwen2 的 norm epsilon 和 RoPE scaling 仍需核对。不得跳过检查直接宣称完全对齐。

## 3. 接手先做的静态工作（不占 GPU）

1. 在独立工作目录拉取交接仓库。改动推送到 `h20/` 开头的新分支，不覆盖旧实验或服务器已有他人目录。
2. 核对材料哈希、完整源码及许可证。已有修改版源码优先于 `reference/project_custom_snapshot/`；后者只用于定位已有机制和差异，不能盲目覆盖更新的源文件。
3. 建立两套隔离 Python 环境，选择双方兼容的同一 Torch/CUDA 基础版本；先查看已有环境可否复用，不反复重装。保留精确 pip freeze、源码 commit/dirty hash。
4. 导入真实训练入口、搜索入口、Profile 入口，不只导入顶层包。尤其检查 `rich`、SciPy、Hydra、OmegaConf、Pydantic、einops、tokenizer 依赖、FA 的 dropout-layer-norm 扩展和 Galvatron DP 扩展。
5. 两系统 CPU 模型配置导出逐项对齐：层数、hidden/FFN、Q/KV、head_dim、vocab/padded vocab、RMSNorm epsilon、RoPE 类型和 scaling、bias、权重绑定、Q/K RMSNorm、dropout、dtype、GBS、优化器、初始化/学习率设置。Qwen3 不得用 LayerNorm 替代 Q/K RMSNorm。
6. 验证 tokenizer 和数据：词表、token ID 上限、padding、数据 hash。不要将 HF 新 tokenizer 与旧 `.bin/.idx` 混用。旧 Qwen2/Qwen2.5 使用 qwen3 tokenizer/data 是已知 benchmark 设置。
7. 修改 H20 启动配置：单节点 8 进程，路径从环境/CLI 输入，不继承 5090 module/账户/节点/HCA/bond0。裸机若无 Slurm，用当前环境的 `python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8`；有调度器则尊重真实分配，不越权占卡。

只读盘点命令（不要清空别人的 GPU 或环境）：

```bash
uname -m
nvidia-smi
nvidia-smi topo -m
command -v python
python --version
command -v nvcc && nvcc --version
command -v conda && conda env list
```

`nvidia-smi` 中 CUDA Version 不是已安装 toolkit 版本。不要把所有包自动升级到最新版。

## 4. 阶段 A：八卡训练验证

先检查单 GPU 的 BF16/FP16 matmul、FlashAttention MHA/GQA/RMSNorm、使用到的 TE/Apex CUDA 内核，再测八卡 allreduce 正确性。随后两系统分别运行小模型、真实 native loop 的八卡训练：

- 至少覆盖 TP4×PP2×DP1（SP 随 TP），验证流水线与 TP；再覆盖一次包含 DP>1 的合法切分，以验证梯度同步。小模型保持足够层数，不把非法切分失败当后端故障。
- 小模型使用随机 token 只作环境验证；它不能替代正式八个模型的数据输入和完整层数。
- 每个验证任务 10 个真实 iteration，检查有限 loss、完成 backward/update、8 个不同 rank 与 GPU 的映射、进程退出、无 skipped iteration。不要只找 “Start training” 或 exit=0。
- 原生 Galvatron 的 runtime profiler 可能改变终止/记录行为；沿用参考中的 native loop 观测方式，确保十步确实发生，不能通过 mock profiler 伪造训练成功。
- 为 Triton/Inductor 等提供相互独立的运行缓存目录，记录环境；不要复制旧 cache、删除运行中任务的 cache，也不要偷偷给某个系统加特殊 allocator 优惠。

两个系统都通过才进入正式实验。某一系统失败时先修环境/包装器；只能做保持语义的兼容修复，保留 patch 和原因，不替换该系统核心搜索算法。

## 5. 阶段 B：八模型公共空间

### 5.1 候选定义

- 单节点 8 GPU，`DP×TP×PP=8`。
- TP、PP 从 `{1,2,4,8}` 取值；MBS 从 `{1,2,4,8}` 取值，保留实际被枚举的每一个值。
- 层数整除 PP；hidden、Q heads、KV heads 整除 TP。
- `GBS % (DP×MBS) == 0`，microbatch 数量 `GBS/(DP×MBS) >= PP`。
- 均匀 PP；TP>1 时启用对应 SP，表达式中的 SP degree=TP，关闭 SP 时分母仍为1，不是0。
- 原生 KV heads 固定；CP/UP/EP=1；recompute、distributed optimizer/ZeRO/FSDP、VPP、offload 等额外优化关闭。

`scripts/common_space.py` 生成**显存剪枝前**的公共清单：依次 40、40、40、40、24、32、40、40，总计296。该数字只是静态结构合法数量，不是已训练成功或显存可行数量。

两系统适配器必须真正消费同一清单，并在实际运行中导出检查/评估记录。仅比较两个配置文件相同不够。集合 hash 之外，另保存模型/数据/优化器/硬件/计时协议的 hash。

### 5.2 显存和 Profile

- 用现场 H20 实际总显存及一致的 headroom 定义公共容量预算，写入协议；不要继承 28 GiB，也不要未经核对固定 96 GiB。
- 各系统可用自己的内存模型和剪枝规则，但不得改变输入候选集合。分别报告各自可行数和拒绝数。
- Galvatron 原生校准可能需要公共清单之外的校准形状；允许其原生模型所需的校准，实测耗时照计。校准 batch 需满足校准时的 world/DP/chunks 整除，不能照搬原 DP16 所用值。
- 任何校准配置 OOM，记录为 Profile 失败，不应伪造成功数据。若确认是包装器的参数错误，修复后使用新 run 或严格记录恢复边界。
- 观察 PyTorch allocated 不是整卡显存真值；区分 allocated/reserved、NVML 整卡占用、其他进程占用。NVML 采样也可能漏瞬时峰值，最终训练是否 OOM 需真实记录。

### 5.3 计时和缓存

- 两系统在 H20 重新实测通信并校准各自模型。项目继续使用原来的单一通信模型及已有接口；本任务不要求重构成拓扑/组规模索引模型。
- 保存 NCCL-tests 原始数据、命令、消息范围、dtype、卡数、NCCL 版本，以及 Galvatron 原生硬件校准结果。H20 测量卡数和执行协议需明确记录，双方最终使用同一台8卡机器；不要把旧平台数值带入。
- 主指标为：**计算/内存 Profile 启动与测量、结果处理、候选搜索这些阶段的实测墙钟合计**。通信校准、排队、环境安装和最终选中配置训练排除，单独报告。
- Devastator 的内部 evaluator/search_seconds 单列，不能与另一系统包含进程启动的阶段墙钟直接做比值。
- 主表按每个 workload 冷计算/内存证据启动；通信证据可预先固定。运行内允许正常复用。公共/完整两阶段不能让后一阶段免费继承前一阶段计算证据，却仍称冷启动公平比较。可额外做双方明确且一致的 warm 口径。
- 禁止用平均算子耗时×查询次数替代实际 Profile/搜索墙钟。

### 5.4 最终训练

获选配置执行 1 次、10 个 iteration，统计6–10。保存每步时间，采用同步完成的整步时间，分布式日志重复行要去重，不能把八个 rank 同一迭代当八个样本。模型/GBS 固定后报告 samples/s 和 tokens/s；公式及单位明确。

只训练选出的最优配置，而非对全部296候选逐一独立训练。winner OOM/失败如实保留；不能无记录地换成第二名并把它写成第一次搜索获胜。

## 6. 阶段 C：八模型完整空间

### 6.1 定义与实现检查

保持同一基准模型、精度、数据和 GBS。2026-09-27 用户修订：Devastator 完整空间开放 CP、UP 和 GQA；GQA 最小可选 KV heads 为 `min(8, 脚本原生 KV heads)`，最大为完整 MHA heads（Q heads），在区间中枚举整除 Q heads 且满足 TP/UP 后端约束的值。该范围允许改变注意力结构，结果必须保存实际 KV heads，不能将全部收益称为同模型语义保持的并行优化收益。公共空间继续固定原生 KV heads。

Devastator 在公共空间上扩展 recompute、distributed optimizer、合法 VPP、CP、UP 和上述 GQA。GPU 并行满足 `DP×PP×TP×CP×UP=8`；UP 不再固定 1，Q/KV 头数须满足 TP×UP 切分，序列须满足原生 CP 两块分割要求。GPU 训练入口将纯 CP 映射为 TE p2p、纯 UP 映射为 a2a、混合映射为 a2a+p2p（层级顺序 `[UP, CP]`）。offload 仍未确认有可搜索实现。

参考 ladder 提供 P0→recompute→distributed optimizer→VPP 的旧实现，但仅是四卡八层先导，需参数化为八卡/八模型，SP=TP、MBS保留四值、1×10后五步。不要无意保留旧的 3 次重复、warm seed 或只测 MLP 的假“全 recompute”描述。

Galvatron 完整空间另走其原生支持的搜索维度和策略组合；公共空间 runner 中的 disable checkpoint/FSDP/CP 等开关需要按原生能力审查，而不是直接解除所有开关。先做有效配置导出和小样本训练验证，再扩大搜索。

每个系统输出 `space_definition.json`：全部维度、取值、固定项、约束、未实现项和后端不支持项。不允许把实际未启用的优化写进完整空间。完整空间候选数可以不同，应称原生/扩展空间系统级对比，不能称同空间机制消融。

GQA/KV heads 搜索按用户修订纳入 Devastator 完整空间；公共空间仍固定原生 KV heads。完整空间结果记录结构变化，不从 10 步训练推断模型质量等价。Galvatron 的结构搜索能力不因该修订自动假定存在。

### 6.2 推荐运行与汇总

先完成八例公共空间，再完整空间，遇到无可行候选保留行。双方各自完整枚举或原生搜索终止规则需冻结并报告；不要临时为某一系统改变超时。当前未指定统一时间预算数值，若采用时间上限实验，须在开跑前记录双方相同预算及到时 incumbent，不能事后挑预算。

Devastator 公共空间应能作为完整空间子集验证；若实现限制造成不是子集，解释并修正命名/协议。若原生 Galvatron 流程另有限制，保留其实际行为。首要交付是两系统×两空间×八例状态表（32行）；不是要求32行都成功。

同一平台另可复用已有 Full/No-Inc/No-Profile/Naive 机制框架，但它不是此轮必须额外重跑的任务，不为了交接擅自扩大 GPU 预算。

## 7. 结果格式、恢复与失败

为每个实验保存：run_id、case_id、system、space、源码 commit/hash、模型/协议/输入 hash、硬件环境、候选清单/hash、真实筛查/评估数量、可行/拒绝数、各阶段墙钟、Profile query/unique key/cache miss 计数（可得时）、winner/effective config、十步时间/loss、测量6–10的均值、状态/首个异常。

状态至少区分：`preflight_failed`、`environment_failed`、`profile_failed`、`search_failed`、`no_feasible_candidate`、`winner_oom`、`training_failed`、`completed`、`timed_out`。失败无有效吞吐，不填0伪装实测，不把基础设施错误直接算成算法无法训练。

每个 run 新目录；只有源码、协议、输入、证据一致且完成记录可验证时跳过已完成项。修改模型/空间/库版本/硬件预算后不可复用旧完成标记。保留原始日志但推送 GitHub 前剔除 token/环境凭据和不必要账户信息；大型原始数据留服务器，上传摘要和 hash。

八个模型都至少有明确结果状态、双方配置核对通过、真实训练与计时证据可追溯，才称这一轮实验完成。不能预先把“原5例可跑、3例不可跑”继承为 H20 结果。

## 8. 交付给用户

每次阶段完成即说明：通过了什么、实际运行了什么、下一步是什么，不让用户反复问。最终提交八模型×两系统×两空间的 CSV/JSON/中文说明、必要图表数据、源码修改和复现命令。只依据实测填写，不预写论文结论。
