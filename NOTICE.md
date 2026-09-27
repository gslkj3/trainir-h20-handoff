# 来源与使用范围

此仓库由项目作者要求整理，用于 TrainIR/Devastator 与 Galvatron 的 H20 实验交接。

- 项目测试脚本及交接文档保留其来源和修改记录，见 source_manifest.json。
- `reference/project_custom_snapshot/megatron/training/initialize.py` 来自作者使用的 Megatron-LM 修改版，保留 NVIDIA 版权。该文件头没有 Apache-2.0 标识；上游默认 NVIDIA 许可条款附于 `licenses/Megatron-NVIDIA.txt`，完整迁移仍应核对实际源版本 LICENSE。
- Galvatron/FlashAttention/NCCL 等依赖仍适用各自上游许可证；本仓库的参考和说明不替换第三方许可，也不声明拥有其著作权。
- 取得完整源码时须同时保留源项目 LICENSE/NOTICE；本仓库未附带完整第三方项目、模型权重、tokenizer 或训练数据。
- `reference/` 文件是迁移参考，不是 H20 正式入口；历史路径已部分脱敏，不应原样执行。
