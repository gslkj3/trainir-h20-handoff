"""Generate a Chinese evidence-linked status report from measured campaign tables."""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from h20_campaign import DEFAULT_ROOT

root=DEFAULT_ROOT
rows=json.loads((root/'results.json').read_text())
pairs=json.loads((root/'paired_results.json').read_text())
assert len(rows)==32 and len(pairs)==16
def number(value):
    return '—' if value is None else f'{value:.3f}'
def link(name):
    return f'[{name}]({name})'
counts=Counter(x['status'] for x in rows)
lines=['# 八模型八卡 H20 对比记录','',
    '更新时间：'+datetime.now(timezone.utc).isoformat(),'',
    '当前 32 行状态：'+ '；'.join(f'{k}={v}' for k,v in sorted(counts.items()))+'。',
    '状态来自 '+link('results.csv')+'；成功训练、终止失败和两空间成对验收分别见 '+
    '、'.join(link(x) for x in ('completion_audit.json','terminal_result_audit.json','common_phase_audit.json','full_phase_audit.json'))+'。', '',
    '每项获选配置训练 1 次、10 步，以下实测为第 6–10 步逐步取八卡最大耗时再求平均。未完成训练的指标为空，不能用预测值填补。',
    '搜索端到端时间包含计算/显存 profiling、处理、初始化、搜索和获选配置准备；提前采集的通信及获选训练排除。主表同时保存全部正式失败尝试的累计搜索时间。首四项历史边界重建均有方法标签。', '',
    '公共空间双方实际候选相同，八模型共 296 项；原生 KV 固定。完整空间分别使用各自能力：Devastator 开放 CP/UP/GQA、优化器分片、重计算和 VPP，Galvatron 使用逐层 TP/Ulysses、DDP/Zero3、checkpoint 与词表层策略。Galvatron 本版原生导出无法保留 CP，固定 CP=1。Devastator GQA 下界 min(8,原生 KV)，上界 Q heads。',
    '完整空间前三个 LLaMA 模型获选 KV 可与原模型不同；这些吞吐比不能解释为同等模型质量收益。全部实验只验证训练执行和性能，没有完成模型质量评估。','']
for space,label in [('common','公共空间'),('full','完整空间')]:
    lines += ['## '+label,'',
        '| 模型 | Devastator 状态 | Galvatron 状态 | Dev 训练秒/步 | Galv 训练秒/步 | Dev 搜索秒 | Galv 搜索秒 | Dev / Galv KV heads |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for pair in pairs:
        if pair['space']!=space:continue
        values=[pair['case'],pair['devastator_status'],pair['galvatron_status'],
            number(pair.get('devastator_mean_iteration_s')),number(pair.get('galvatron_mean_iteration_s')),
            number(pair.get('devastator_search_e2e_seconds')),number(pair.get('galvatron_search_e2e_seconds')),
            f"{pair.get('devastator_kv_heads', '—')} / {pair.get('galvatron_kv_heads', '—')}"]
        lines.append('| '+' | '.join(values)+' |')
    lines.append('')
lines += ['## OOM 与失败口径','',
    '主指标只计获选配置正式训练的实际 CUDA OOM；原生 profiling 失败单独列出，预测无可行候选不计为实测 OOM。',
    'Qwen2-7B 公共空间的 Devastator 无可行候选表示模型拒绝了 32 个候选，不证明这 32 项实测全部 OOM。',
    'Qwen3-14B 完整空间 Devastator/Megatron 的原始获选训练在第 3 步发生 OOM：请求 1.45 GiB，设备空闲 1.40 GiB，PyTorch 已分配 76.50 GiB，已保留但未分配 12.91 GiB。这些是异常打印的舍入值。',
    'Galvatron Qwen2-1.5B/7B 早期原生显存 profiling OOM 已留存；后续采用原生 sequence profiling 模式恢复工作流，目标训练序列不变。Profile 修复并不能证明原失败由碎片化引起。',
    '事件证据见 '+link('oom_incidents.json')+'，统计口径见 '+link('oom_reporting_policy.json')+'。','']
diagnostic=root/'diagnostics/allocator-expandable-01'
if (diagnostic/'status.json').exists():
    state=json.loads((diagnostic/'status.json').read_text())
    lines += ['分配器对照状态：`'+state['status']+'`。仅调整 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，其余工作负载与原始源码快照相同，输出隔离。',
        '若通过，将对应 OOM 归为内存管理敏感问题；原始失败不删除，诊断性能不静默替换默认分配器主表。']
    if (diagnostic/'results.json').exists():
        for item in json.loads((diagnostic/'results.json').read_text()):
            lines.append(f"- `{item['job']}`：{item['status']}，{item['classification']}；[证据]({Path(item['evidence']).relative_to(root)}/resolution.json)。")
    lines += ['', '独立对照验收见 '+link('allocator_diagnostic_audit.json')+'；通过验收的训练性能单列 '+link('allocator_diagnostic_metrics.csv')+'，不混入默认分配器主表。']
    audit_path=root/'allocator_diagnostic_audit.json'
    if audit_path.exists():
        audited=json.loads(audit_path.read_text())['records']
        if any(x['diagnostic']=='megatron_selected_training' and x.get('verified') and x['status']=='passed' for x in audited):
            lines += ['', '正式训练 OOM 归因：Qwen3-14B 完整空间原获选配置仅调整分配器后完成十步，并通过独立审计，记录为内存管理问题，不计为已证实的固有显存预测错误。公共空间正式训练 OOM 0 次；完整空间历史正式训练 OOM 1 次，其中 1 次经分配器调整解决，未解决 0 次。原始 OOM 和 12.91 GiB 未使用缓存仍保留。']
            metrics=json.loads((root/'allocator_diagnostic_metrics.json').read_text())
            for metric in metrics:
                lines.append(f"修复后的独立训练：`{metric['case']}`，第 6–10 步均值 **{metric['mean_iteration_s']:.3f} 秒/步**，分配器 `{metric['allocator']}`。")
lines += ['', '## 绘图与重放','',
    '逐步和逐卡实测：'+link('steps.csv')+'、'+link('rank_steps.csv')+'；成对比值、KV 差异与搜索时间：'+link('paired_results.csv')+'；显存峰值和失败原因：'+link('results.csv')+'。',
    '搜索候选：'+link('candidates.csv')+'；Profile 明细：'+link('profile_accounting.csv')+'；Galvatron 逐层获选策略：'+link('galvatron_selected_layers.csv')+'。字段单位和口径见 '+link('table_schema.json')+'。',
    '命令、环境、源码快照、配置和日志哈希见 '+link('replay_index.json')+'。重放必须使用新 attempt/retry 目录，避免历史输出路径覆盖。安装与启动方式见仓库 `docs/H20_CAMPAIGN_CN.md`；通信原始记录见 `hardware/`；私有输入哈希见 `input_receipt.json` 和 `input_audit.json`。',
    '历史留存限制：最早 30 个阶段没有逐阶段源码快照，但保留运行命令、配置、日志和流程启动时的脚本哈希。'+link('source_recovery_index.json')+' 提供哈希相同的现存源码引用及缺失清单；不能把启动时哈希当作全部后续阶段源码的证明，也不能声称这些早期阶段均可精确恢复历史源码。',
    '每秒 GPU 采样保留在各 stage 的 `.gpu.csv`；采样峰值可能漏掉瞬时峰值，Torch allocated/reserved 与 NVML 整卡用量不可混为一项。']
(root/'REPORT_CN.md').write_text('\n'.join(lines)+'\n')
print('Wrote',root/'REPORT_CN.md')
