"""Expand native Galvatron selected layouts; these are configurations, not measurements."""
import csv
import json
from pathlib import Path
from h20_campaign import DEFAULT_ROOT, save, sha


def export(root=DEFAULT_ROOT):
    rows=[]
    cases={c['id']:c for c in json.loads((root/'cases8.json').read_text())['cases']}
    for result in json.loads((root/'results.json').read_text()):
        if result['system']!='galvatron' or not result.get('path'):continue
        run=Path(result['path']);files=list((run/'selected').glob('galvatron_config_*.json'))
        if not files:continue
        assert len(files)==1
        config=json.loads(files[0].read_text());case=cases[result['case']]
        arrays={k:list(map(int,config[k].split(','))) for k in ('tp_sizes_enc','use_sp','dp_types_enc','checkpoint')}
        assert all(len(v)==case['num_layers'] for v in arrays.values())
        stages=[i for i,n in enumerate(map(int,config['pp_division'].split(','))) for _ in range(n)]
        assert len(stages)==case['num_layers']
        for layer in range(case['num_layers']):
            degree=arrays['tp_sizes_enc'][layer];ulysses=degree if arrays['use_sp'][layer] else 1
            tp=1 if arrays['use_sp'][layer] else degree
            divisor=config['pp_deg']*tp*ulysses
            assert config['world_size']%divisor==0
            dp=config['world_size']//divisor
            denominator=dp*config['chunks']
            rows.append(dict(case=result['case'],space=result['space'],outcome_status=result['status'],
                run=str(run),selected_config=str(files[0]),selected_config_sha256=sha(files[0]),
                layer_index=layer,pipeline_stage_index=stages[layer],pp=config['pp_deg'],tp=tp,
                ulysses=ulysses,cp=1,dp=dp,checkpoint=bool(arrays['checkpoint'][layer]),
                dp_type='zero3' if arrays['dp_types_enc'][layer] else config['default_dp_type'],
                native_kv_heads=case['native_kv_heads'],global_batch_size=config['global_bsz'],
                chunks=config['chunks'],implied_micro_batch_size=config['global_bsz']/denominator,
                integral_micro_batch=config['global_bsz']%denominator==0,
                vocab_tp=config['vtp'],vocab_sp=config['vsp'],vocab_zero3=bool(config['embed_sdp'])))
    save(root/'galvatron_selected_layers.json',rows)
    with (root/'galvatron_selected_layers.csv').open('w',newline='') as f:
        fields=list(rows[0]) if rows else ['case','space','layer_index']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print('Exported Galvatron selected layer rows:',len(rows))
    return rows


if __name__=='__main__':export()
