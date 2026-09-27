# 来源与使用范围

此仓库由项目作者要求整理，用于 TrainIR/Devastator 与 Galvatron 的 H20 实验交接。

- 项目测试脚本及交接文档保留其来源和修改记录，见 source_manifest.json。
- `reference/project_custom_snapshot/megatron/training/initialize.py` 来自作者使用的 Megatron-LM 修改版，保留 NVIDIA 版权。该文件头没有 Apache-2.0 标识；上游默认 NVIDIA 许可条款附于 `licenses/Megatron-NVIDIA.txt`，完整迁移仍应核对实际源版本 LICENSE。
- Galvatron/FlashAttention/NCCL 等依赖仍适用各自上游许可证；本仓库的参考和说明不替换第三方许可，也不声明拥有其著作权。
- `sources/h20_sources_20260927.tar.gz` 已包含收到的两个修改版项目源码快照及各自完整 LICENSE；各文件原有版权声明保留。Megatron 默认 NVIDIA 条款及其列明的第三方许可、Galvatron Apache-2.0 及其列明的 NVIDIA 第三方许可分别适用。
- 快照中的 `Megatron-LM/run.sh`、`Megatron-LM/test1.sh` 仅在导出副本中替换了 URL 用户名/密码为环境变量占位符；原始与导出 SHA256 见包内清单。旧站点路径和命令尚未适配 H20。
- 本仓库不附带模型权重、tokenizer 或训练数据；扫描和人工检查不是对任意代码安全性或实验可复现性的保证。
- `reference/` 文件是迁移参考，不是 H20 正式入口；历史路径已部分脱敏，不应原样执行。
