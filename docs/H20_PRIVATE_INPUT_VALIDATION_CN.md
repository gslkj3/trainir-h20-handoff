# H20 私有输入接收与阶段 A 验证（2026-09-27）

本次接续 `4d62e0c` 交接，复用 `/opt/hbv/trainir-h20-runtime` 及上一轮 Galvatron 环境；没有覆盖重解源码，也没有替换交付的 Megatron 修改版。原交接工作区 `/opt/hbv/trainir-h20-handoff` 保留，新脚本位于 `/opt/hbv/trainir-h20-handoff-received`。

原始证据：`/opt/hbv/trainir-h20-validation-20260927`。本次是小模型真实输入集成验证，不是八个完整模型的性能、Profile 或搜索结果。观测器带参数克隆、梯度检查及同步开销，日志时间不能直接用于系统性能比较。

## 输入与源码

- 接收前重新核对源码快照中的 2331 个文件。快照 SHA256：`649f9ab881662500ed069317a6c74ff9ab2662dcf975a2f3449ca507cc57766c`。
- 两个 `/workspace` 包中，16 个 tokenizer/config 文件和 6 个数据文件均与交接仓库清单逐文件 SHA256 一致。tokenizer 包没有外部整包哈希，验证依据是仓库逐文件清单；没有把包内 manifest 自证为信任依据。
- 数据包整包 SHA256：`745267ff68b68c8940b32935357eb2f45e1775b269ab1040e1053445708ed170`。
- tokenizer 包实收 SHA256：`98765e4234539932252eab6d5c82ba0f8fe28f7e1b41168982e9a7d1169c34db`。
- 三组 `.idx` 指针、长度、文档边界、`.bin` 大小及全部 token ID 均已核对；ID 均存在于对应旧 tokenizer 中。LLaMA / LLaMA3 / Qwen 分别为 147359171 / 122379171 / 128297270 个 token。
- Qwen tokenizer 实际最大 ID 为 151668，但本轮显式 embedding padded vocab 为 151936；不能从 tokenizer 长度自动推导正式模型词表。
- Galvatron 交付核心与官方 `cea12ffb146a220643c8f99f9cb84294755d29f8` 一致。交付包的包装和实验入口不同，未另造训练算法。

## 环境与依赖

Galvatron 继续使用 `/opt/hbv/venv-galvatron-h20`：Torch 2.9.0+cu128、FlashAttention 2.8.3、独立 dropout_layer_norm CUDA 扩展。其原生 AdamW 和梯度裁剪回退实现不要求 Apex，因此没有为可选依赖重装该环境。

Megatron 使用 `/opt/hbv/venv-megatron-h20`，通过 `.pth` 只读复用上述 Torch/CUDA/FlashAttention，再单独安装：

- NVIDIA Apex 官方源码 `8a6508aaad6e75a2b939e33f308cd63d745d97f1`，启用 `APEX_CPP_EXT=1 APEX_CUDA_EXT=1 TORCH_CUDA_ARCH_LIST=9.0` 本机编译。
- Transformer Engine 2.9.0，遵循交付 Megatron 的 `<2.10` 依赖约束；PyTorch 扩展本机编译。TE 原生 RMSNorm 支持本轮需要的 sequence parallel。
- pandas、scikit-learn、ONNX 等交付入口所需 Python 依赖。保留 NumPy 1.26.4，未升级基础 Torch/CUDA 栈。

TE 首次编译缺 `cudnn.h`，通过 `CPLUS_INCLUDE_PATH` 指向现有 `site-packages/nvidia/cudnn/include` 修复。运行时还需：

```bash
export CUDNN_HOME=/opt/hbv/venv-galvatron-h20/lib/python3.10/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}"
```

新启动器自动设置这些变量。不能删除被共享的 Galvatron 环境，否则 Megatron 的共享依赖也会失效。

`cuda-dependencies.json` 记录 Apex/TE LayerNorm、RMSNorm，FlashAttention GQA 的 FP16/BF16 前反向与 PyTorch 参考对照，以及 Apex FusedAdam 10 步对照 AdamW，全部通过。Megatron 环境 `pip check` 通过；八卡 NCCL 的 FP32/FP16/BF16 allreduce 及半精度 matmul backward 通过。构建日志和 Apex、TE Torch wheel 均保留。

## 配置修复与运行方式

发现 Galvatron 模型文件加载器保留部分非空 schema 默认值，使文件中的 `rotary_base`、`qk_layernorm` 没有生效。启动器现在将所需字段显式放入 `runtime.model`，并检查解析后的配置。首轮 LLaMA3/Qwen 训练虽然完成，但配置审计失败，保留记录并由 `-02` 运行替代；不能引用首轮作为正确模型配置的证据。

Qwen 的 Q/K norm 在启动器中绑定到后端已有 `GalvatronNorm`（实际返回 FlashAttention RMSNorm），避免默认的 LayerNorm。没有替换原生训练循环、优化器或 profiler。该绑定和配置断言保存在 `h20_galvatron_validation.py`。

共同小模型：4 层、hidden 512、FFN 1376、Q 8/KV 4、head_dim 64、seq 128、GBS 8、PP 2、SP 开启。LLaMA 使用 FP16；LLaMA3/Qwen 使用 BF16。Qwen 为缩小模型的结构/输入验证，不等于完整 Qwen3-14B。

```bash
python3 scripts/run_h20_input_validation.py \
  --system megatron --case llama-tp4-fp16 \
  --out /opt/hbv/trainir-h20-validation-20260927/new-unique-run
```

`--system` 可选 `galvatron` / `megatron`；`--case` 可选 `llama-tp4-fp16`、`llama-tp2-fp16`、`llama3-tp4-bf16`、`qwen3-tp2-bf16`。输出目录必须不存在，失败记录不会覆盖。每个运行保存命令、包版本、原始日志、逐 rank 有效配置及逐步更新证据。

完整八模型的 CPU 有效配置等价审计仍需另做，特别是 Qwen2 epsilon / RoPE scaling、模型初始化和优化器精度语义；本轮通过不代表正式实验全部对齐。

## 最终结果

| 输入 / 精度 / 并行 | Galvatron | Megatron |
|---|---|---|
| LLaMA / FP16 / TP4×PP2×DP1 | 通过（01） | 通过（01） |
| LLaMA / FP16 / TP2×PP2×DP2 | 通过（01） | 通过（01） |
| LLaMA3 / BF16 / TP4×PP2×DP1 | 通过（02） | 通过（01） |
| Qwen / BF16 / TP2×PP2×DP2 | 通过（02） | 通过（01） |

八组均为 8 个不同 GPU/rank、每 rank 10 次真实参数更新、有限 loss、正常退出。Megatron 原生日志每组 10 步均无 skipped iteration；Galvatron 每步都观测到真实参数变化。DP2 运行的梯度副本抽样一致性通过。64 份 rank 有效配置逐项核对小模型维度、词表、RMSNorm、RoPE 和 Q/K norm 开关。Megatron Qwen 另记录了实际 Q/K RMSNorm 模块类型。

摘要和输入/环境审计见 `reports/h20-input-validation-20260927/`；完整日志、逐 rank 记录、独立编译缓存留在外部原始证据目录，不把私有数据或 tokenizer 内容提交到 Git。现有交接测试 29 项通过。

阶段 A 本轮验证完成；阶段 B/C 尚未运行，不输出正式吞吐或搜索收益结论。
