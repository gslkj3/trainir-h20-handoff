"""Verify native no-candidate and actual selected-training OOM outcomes separately."""
import json
import math
from datetime import datetime
from h20_campaign import DEFAULT_ROOT, save, sha
from h20_workloads import cases
from common_space import candidates, require_equal


def audit(root=DEFAULT_ROOT):
    records=[]
    for c in cases(root).values():
        for system in ('devastator','galvatron'):
            for space in ('common','full'):
                attempts=sorted((root/'runs'/c['id']/system/space).glob('*'))
                if not attempts:continue
                run=attempts[-1]
                if not (run/'status.json').exists():continue
                status=json.loads((run/'status.json').read_text())['status']
                classification_path=run/'failure_classification.json'
                if status=='failed' and classification_path.exists():
                    classification=json.loads(classification_path.read_text())
                    if classification['classification']=='selected_training_cuda_oom':
                        record=dict(case=c['id'],system=system,space=space,run=str(run),verified=False)
                        try:
                            from h20_audit_selected_oom import audit_selected_oom
                            record.update(audit_selected_oom(root,run,c,system,space))
                        except Exception as exc:record['issue']=repr(exc)
                        records.append(record)
                if status!='no_feasible_candidate':continue
                record=dict(case=c['id'],system=system,space=space,run=str(run),verified=False)
                try:
                    stage=json.loads((run/'stages/search.json').read_text())
                    assert stage['status']=='completed' and stage['returncode']==0
                    assert sha(run/'stages/search.log')==stage['log_sha256']
                    timing=json.loads((run/'search_timing.json').read_text())
                    assert timing['outcome']=='no_feasible_candidate'
                    assert math.isfinite(timing['search_e2e_seconds']) and timing['search_e2e_seconds']>=stage['seconds']
                    if system=='devastator':
                        result=json.loads((run/'search.json').read_text())
                        assert result['best'] is None and not result['evaluated']
                        assert result['rejected'] and all(x['predicted_feasible'] is False for x in result['rejected'])
                        if space=='common':
                            observed=[tuple(x['parallel'][i] for i in (0,1,4,7)) for x in result['rejected']]
                            require_equal(observed,candidates(c),'no-candidate common coverage')
                        else:
                            raise AssertionError('Full-space terminal coverage audit still required')
                        contract=json.loads((run/'contract.json').read_text())
                        comm=root/'hardware/devastator/comm_data/profile_comm.json'
                        assert sha(comm)==contract['communication_sha256']==sha(run/'native_evidence/comm_data/profile_comm.json')
                        assert comm.stat().st_mtime<datetime.fromisoformat(stage['start_utc'].replace('Z','+00:00')).timestamp()
                        record.update(rejected_candidates=len(result['rejected']),reason_counts={reason:sum(x['reason']==reason for x in result['rejected']) for reason in {x['reason'] for x in result['rejected']}})
                    else:
                        raise AssertionError('Galvatron terminal coverage audit still required')
                    assert not (run/'training_summary.json').exists()
                    record.update(verified=True,interpretation='Native search rejected every declared candidate. This is a cost-model outcome, not proof that every configuration physically OOMs.',search_e2e_seconds=timing['search_e2e_seconds'])
                except Exception as exc:
                    record['issue']=repr(exc)
                records.append(record)
    save(root/'terminal_result_audit.json',dict(records=records,note='Verifies explicit native no-candidate or selected-training OOM outcomes; does not substitute for successful training or complete the campaign.'))
    return records


if __name__=='__main__':
    records=audit()
    print(dict(terminal_rows=len(records),verified=sum(x['verified'] for x in records)))
