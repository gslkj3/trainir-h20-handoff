"""Summarize retained evidence; never infer training success from exit code alone."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('results', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
rows = []
for folder in sorted(a.results.iterdir()):
    if not (folder/'status.json').exists():
        continue
    status = json.loads((folder/'status.json').read_text())
    launch = json.loads((folder/'launch.json').read_text())
    ranks = [json.loads(f.read_text()) for f in sorted(folder.glob('rank[0-7].json'))]
    audit = folder/'configuration-audit.json'
    config_ok = not audit.exists() or json.loads(audit.read_text())['passed']
    row = dict(run=folder.name, **status, accepted=bool(status['passed'] and config_ok),
        tp=launch['tp'], pp=launch['pp'], dp=launch['dp'], precision=launch['precision'],
        rank_count=len(ranks), config_audit_passed=config_ok,
        training_log_sha256=hashlib.sha256((folder/'train.log').read_bytes()).hexdigest())
    if ranks:
        row['updates_per_rank'] = [len(r['updates']) for r in ranks]
        row['minimum_changed_elements'] = min(u['changed_elements'] for r in ranks for u in r['updates'])
        row['dp_gradient_replica_checks'] = sum(u['dp_gradient_replica_checks'] for r in ranks for u in r['updates'])
        model = ranks[0]['effective']
        if status['system'] == 'galvatron':
            model = model['model']
        row['effective_model'] = {k:model.get(k) for k in ('num_layers', 'hidden_size',
            'ffn_hidden_size', 'num_attention_heads', 'num_query_groups', 'kv_channels',
            'padded_vocab_size', 'normalization', 'norm_epsilon', 'rotary_base', 'qk_layernorm')}
    rows.append(row)
report = dict(scope='Phase A: eight-GPU small-model real-input integration, not eight full-model benchmarks.',
    results_root=str(a.results.resolve()), runs=rows,
    accepted_runs=sum(r['accepted'] for r in rows),
    limitations=['Instrumented timing is not performance data.',
                 'Full eight-model effective-configuration parity and profile/search are pending.'])
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'runs'}))
