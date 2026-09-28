"""Verify full-space paired outcomes, preserving native search/selected-training failures."""
import json
from h20_campaign import DEFAULT_ROOT, save, sha
from h20_workloads import cases

root=DEFAULT_ROOT
success=json.loads((root/'completion_audit.json').read_text())['rows']
terminal=json.loads((root/'terminal_result_audit.json').read_text())['records']
verified={x.get('run'):x for x in success+terminal if x.get('verified')}
results=json.loads((root/'results.json').read_text())
pairs=[]
for case in cases(root).values():
    record=dict(case=case['id'],verified=False,systems={},issues=[])
    try:
        for system in ('devastator','galvatron'):
            matches=[x for x in results if (x['case'],x['system'],x['space'])==(case['id'],system,'full')]
            assert len(matches)==1
            row=matches[0]
            assert row['status'] in ('completed','failed','no_feasible_candidate'),row['status']
            assert row['path'] in verified,'Outcome lacks verified evidence'
            evidence=verified[row['path']]
            if row['status']=='failed':
                assert evidence.get('classification')=='selected_training_cuda_oom'
            record['systems'][system]=dict(run=row['path'],status=row['status'],
                classification=evidence.get('classification'),
                measured_training=row['status']=='completed',
                search_e2e_seconds=row.get('search_e2e_seconds'),
                kv_heads=row.get('kv_heads'))
        record['verified']=True
    except Exception as exc:
        record['issues'].append(repr(exc))
    pairs.append(record)
save(root/'full_phase_audit.json',dict(all_eight_pairs_verified=all(x['verified'] for x in pairs),
    verified_pairs=sum(x['verified'] for x in pairs),pairs=pairs,
    audit_inputs={name:sha(root/name) for name in ('results.json','completion_audit.json','terminal_result_audit.json')},
    note='Paired outcome audit, not a claim of all successful training or equal full spaces. Allocator diagnostics remain separately labelled. Underlying row audits validate actual spaces, selected configs, timing and execution evidence.'))
print('Verified full outcome pairs:',sum(x['verified'] for x in pairs),'/8')
