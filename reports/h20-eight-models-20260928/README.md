# H20 八模型实验结果（2026-09-28）

八模型 × Devastator/Megatron 与官方 Galvatron × 公共/完整空间，共 32 行结果已验收。

- 30 项在原运行设置下完成十步训练。
- Qwen3-14B 完整空间 Devastator 原获选配置发生 OOM；仅设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后十步通过，独立审计归为内存管理问题。修复后的均值为 61.740 秒/步。
- Qwen2-7B 公共空间 Devastator 搜索无可行候选，保留该结果，不视为实测 OOM。
- Galvatron Qwen2-7B 历史显存 profiling OOM 在相同分配器调整后也通过，单独统计，不混入正式训练 OOM。

## 阅读入口

- [中文报告](REPORT_CN.md)
- [32 行结果](results.csv) / [JSON](results.json)
- [16 组成对比较](paired_results.csv)
- [分配器修复后的独立性能](allocator_diagnostic_metrics.csv)
- [OOM 事件及归因](oom_incidents.csv)
- [完成审计](goal_completion_audit.json)
- [字段与单位](table_schema.json)
- [复现命令与证据索引](replay_index.json)
- [运行协议和操作说明](../../docs/H20_CAMPAIGN_CN.md)

搜索时间包含计算/显存 profiling、处理、初始化和搜索，仅排除预采集通信与获选训练。训练性能统计第 6–10 步。完整空间可能改变 KV heads，性能比不等于模型质量等价。

## 发布范围与证据定位

本目录公开完整绘图表、结果和审计元数据；脚本、协议与补丁位于仓库相应目录。JSON/CSV 保留原始内容及哈希，绝对路径指向原 H20 服务器，不是 GitHub 上的下载路径。原始逐阶段日志、显存采样、源码快照保留在服务器 `/opt/hbv/trainir-h20-experiments/20260927-v1`；此发布不声称已上传这些全部原始文件。

私有输入只发布哈希和检查结果，不发布数据、tokenizer 文件或权重。重放需要自行提供获授权的相同输入及八卡 H20 环境，并设置本地路径。

历史限制：最早 30 个阶段没有独立源码快照；其命令、配置、日志和流程启动哈希保留，见 [哈希恢复索引](source_recovery_index.json)。其余 430 个正式阶段有快照。`PUBLICATION_MANIFEST.json` 记录本目录复制的原始结果文件哈希。
