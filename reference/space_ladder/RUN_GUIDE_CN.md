# 四卡内部空间递增：先导实验 v1

## 文件与目录

只新增目录 `Megatron-LM/dtsir_space_tests/`，将 run_ladder.py、space_entry.py、test_ladder.py 放进去。无需覆盖 test_parallel_model.py、dtsir_collect.py、initialize.py 或 dtsir_ir_tests 中的任何文件。依赖当前已跑通的 IR 测试版本及其初始化钩子。

脚本会读取当前成功使用的 dtsir_ir_tests/MM_llama_7b.sh，保留数据/tokenizer/backend 参数，将训练入口替换为本包的 space_entry.py，并在结果目录写入 launcher.sh。不是重新使用包内旧的数据路径。不修改源启动脚本。

## 范围

- 单节点四 GPU、8 层、hidden 4096、FFN 11008、32 注意力头、seq 2048、GBS 256、原启动脚本精度与后端。
- P0：DP/TP/PP；P1：增加 selective MLP 重计算；P2：增加优化器状态分片；P3：增加 VPP。
- 每种新增策略都有关闭选项。VPP 使用每虚拟阶段 1 或 2 层并过滤不可整除和无效 PP 组合。
- 四级均固定 SP 关闭，CP/UP/EP=1；四级均搜索 microbatch 1、2（可通过 --mbs 1 2 4 等显式扩展，必须新结果目录）。这不是把 microbatch 固定为 1，也不是声称已覆盖全部重计算方式/所有策略空间。
- 默认固定 KV 头数32，即本次与刚完成的8层实验保持相同模型；若需要固定 GQA8，在新结果目录加 --kv-heads 8，四级模型结构仍相同。不搜索 GQA。
- 这是先导实验。正式6–8用例的规模、资源和模型参数要另行配置，不要简单改掉模型检查就当正式全空间实验。

## 可直接执行

在 GPU 节点，确认没有其他测试占用这四张卡，在已经跑通的 py312-t29 环境：

```bash
cd ~/run/wjy/Megatron-LM
conda activate py312-t29
python dtsir_space_tests/test_ladder.py
```

先做 P0（一次搜索、一次观测、三次无观测训练）：

```bash
RUN_DIR="mm_logs/space_ladder_5090_pilot_$(date +%Y%m%d_%H%M%S)"
echo "$RUN_DIR"
nohup python -u dtsir_space_tests/run_ladder.py --launcher dtsir_ir_tests/MM_llama_7b.sh --out "$RUN_DIR" --seed-evidence mm_logs --mbs 1 2 --through 0 > "${RUN_DIR}.driver.log" 2>&1 &
tail -f "${RUN_DIR}.driver.log"
```

看到 DONE 后按 Ctrl+C 退出 tail，查看：

```bash
cat "$RUN_DIR/ladder_summary.md"
```

先把 summary 和日志发回确认。确认后执行同一命令，将 --through 0 改为 --through 3；使用同一个 RUN_DIR，P0成功结果将跳过。

```bash
nohup python -u dtsir_space_tests/run_ladder.py --launcher dtsir_ir_tests/MM_llama_7b.sh --out "$RUN_DIR" --seed-evidence mm_logs --mbs 1 2 --through 3 > "${RUN_DIR}.continue.driver.log" 2>&1 &
tail -f "${RUN_DIR}.continue.driver.log"
```

新终端必须重新设置 RUN_DIR 为原路径，不要重新执行带 date 的赋值。不同时运行两份 runner。不要在实验期间更改核心代码、启动脚本或 mm_logs/calc_data、comm_data；续跑会校验这些输入。

## 缓存和统计

使用冻结的已有5090 Profile/通信校准快照（同硬件/软件环境）；每级有独立副本。缺失键仍由当前硬件测量。不把前一级新增缓存传给后一级，不宣称冷启动结果。不要拿A100算子测量作5090种子。每级统计完整搜索耗时、查询/分析/测量计数、筛选/可行候选、全部候选和最优配置。

同一候选集的数学最优预测值在固定代价下应随空间扩大不增加；独立补测可能造成数值变化，因此程序记录 shared-key 成本变化和最优值是否单调，而不伪造一致性。实际吞吐不要求单调。无效查询或校准缺失日志必须发回检查，不按零成本认定成功。

每级 winner 都保留原预测选择；OOM按结果记录，不偷偷换第二名。若观测 OOM，仍尝试 clean 以区分观测影响；clean发生OOM即停止其重复训练，继续下一级。其他异常会停机等待检查。已经完成和已记录OOM任务不会重跑；无status的半截任务也不会自动覆盖。

## 结果

ladder_summary.json/md；P0–P3/search/search.json；每级observe_0和clean_0–2；同一份种子校验信息。训练计时每次10步，去首步，汇总三次独立运行。allocated峰值仅为附带诊断，不计算总显存预测误差。

```bash
tar -czf space_ladder_5090_pilot_results.tar.gz "$RUN_DIR"
```

这里只完成本地静态/逻辑测试，尚未在服务器GPU上跑过该新搜索驱动。先验收P0再推进P1–P3。
