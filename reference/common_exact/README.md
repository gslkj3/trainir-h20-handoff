# 五用例 16 卡公共空间修订包

本包沿用现有的 `dtsir_common16`、`dtsir_galvatron6` 和 `paired16` 测试框架。先备份服务器上同名文件，再按相对路径覆盖。不要将旧实验目录改名或删除；新任务使用独立时间戳标签。

| 本包文件 | 服务器目标（相对于仓库根目录） |
| --- | --- |
| `megatron/dtsir_common16/{run.py,entry.py,common5_space.py}` | Megatron-LM 中的 `dtsir_common16/` |
| `megatron/paired16/{all5_exact.sbatch,submit_exact.sh,preflight_exact.py,collect_exact.py}` | Megatron-LM 中的 `paired16/` |
| `galvatron/dtsir_galvatron6/{run_six.py,common5_space.py,cases.json}` | Hetu-Galvatron-dtsir 中的 `dtsir_galvatron6/` |

`cases.json` 将五个参与比较的用例的 RMSNorm epsilon 显式统一为 Megatron 实际运行时的 `1e-5`；不参与此比较的 Qwen2-1.5B 条目保持原值。

公共候选由 `(DP, PP, TP, MBS)` 唯一确定：16 卡；PP 和 TP 为不超过 8 的 2 的幂；MBS 为 1、2、4、8；层数、hidden、head 与 GQA 的整除约束生效；流水线微批次数至少等于 PP。CP、UP、EP、重计算、VPP、状态切分都固定关闭，TP>1 时使用对应的 TP 序列并行。所有候选使用同一 28 GiB 预测显存上限。五例候选数依次为 52、52、52、48、52。

新任务中，Megatron 禁用已有算子测量种子并计入缺失算子的实测时间；Galvatron 计入计算/显存 Profile、处理结果和搜索的耗时。双方均不把通信硬件校准计入搜索时间。Galvatron 的 RMSNorm epsilon 与 Megatron 有效参数统一为 `1e-5`。显存 Profile 保持全局 batch 16：原生校准包含 DP16，上一版改成 4 不满足全局 batch 的整除约束，现已纠正。这个参数不是候选搜索的 MBS。两边都保留真实训练为 1 次、10 步、统计后 5 步。

汇总默认显示包含启动开销的阶段墙钟：Megatron 使用 `search_stage_wall_s`，Galvatron 使用计算/显存 Profile 启动、处理和搜索的合计；Megatron 原有内部评估时间 `search_seconds` 仍单独保留，不与 Galvatron 包含启动的时间混称同一指标。两者均排除最终训练、排队和通信校准，不能直接视为完整提交到结果的耗时。

候选集合相同不意味着两系统内部 Profile 项目相同：Galvatron 的原生代价模型仍可能要求额外的校准配置，这些实际耗时照实计入。任何 Profile 阶段失败都会记录在对应 `status.json` 中，该用例不会得到吞吐或搜索加速比。

登录节点先检查，不提交、不占用 GPU：

```bash
cd /data/run01/LEGACY_USER/wjy/Megatron-LM
bash paired16/submit_exact.sh --check-only
```

看到 `CHECKS PASSED; no Slurm job submitted.` 后，执行 `bash paired16/submit_exact.sh` 提交。脚本先在登录节点检查五例模型字段、Megatron 历史候选集合与 Galvatron 对这组策略的表达能力，并生成、验证全部原生 Profile 启动参数；全部通过才提交单个 2 节点、16 卡作业，顺序运行两系统各五例。默认生成带时间戳和进程号的新标签，拒绝覆盖已存在的输出目录。若只想先检查 Galvatron 的候选表示及 Profile 命令：

```bash
cd /data/run01/LEGACY_USER/wjy/dependencies/Hetu-Galvatron-dtsir
source dtsir_galvatron6/env.sh
MEGATRON_ROOT=/data/run01/LEGACY_USER/wjy/Megatron-LM python dtsir_galvatron6/run_six.py space-check
```

完成后查看结果，使用提交时打印的 `PAIR16_TAG`：

```bash
cd /data/run01/LEGACY_USER/wjy/Megatron-LM
python paired16/collect_exact.py "${PAIR16_TAG}"
```

提交脚本内导出的变量不会返回父终端，请把实际打印的标签填入上面的命令；也可从 `mm_logs/` 的新目录名查找。`collect_exact.py` 区分预设集合一致和实际评估完成，并核查双方实际候选记录；失败用例只报告状态，不生成加速比。

2026-09-23 文件审计：本地使用所提供固定版本的原生配置定义、策略枚举方法和 Profile 命令生成器，五例均通过；每例生成 2 条计算、28 条显存 Profile 命令。Python 语法、旧 epsilon 配置拒绝、重复候选拒绝均通过。没有在本地运行 CUDA、分布式训练或真实代价模型数值评估；服务器仍需执行 `--check-only`。保持 batch 16 的原生 Profile 仍可能真实 OOM，不保证五例都能完成，也不把脚本检查当成成功训练证据。
