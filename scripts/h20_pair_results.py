"""Plot-ready paired measurements; never substitute predictions for missing runs."""
import csv
import json
from h20_campaign import DEFAULT_ROOT, save


def export_pairs(root=DEFAULT_ROOT):
    results=json.loads((root/'results.json').read_text())
    cases=json.loads((root/'cases8.json').read_text())['cases']
    index={(r['case'],r['space'],r['system']):r for r in results}
    pairs=[]
    for case in cases:
        for space in ('common','full'):
            d=index[(case['id'],space,'devastator')]
            g=index[(case['id'],space,'galvatron')]
            row=dict(case=case['id'],space=space,native_kv_heads=case['native_kv_heads'])
            for system,r in [('devastator',d),('galvatron',g)]:
                for key in ('status','path','training_evidence','kv_heads','search_e2e_seconds',
                            'search_timing_method','search_cumulative_e2e_seconds','search_attempt_count',
                            'search_attempts_timing_complete','memory_profile_mode',
                            'memory_profile_min_sequence','memory_profile_max_sequence',
                            'mean_iteration_s','tokens_per_second',
                            'peak_allocated_bytes','peak_reserved_bytes','peak_nvml_mib'):
                    row[system+'_'+key]=r.get(key)
            complete=all(r['status']=='completed' and r.get('mean_iteration_s',0)>0 for r in (d,g))
            row['both_training_completed']=complete
            row['same_kv_heads']=d.get('kv_heads')==g.get('kv_heads') if all(r.get('kv_heads') is not None for r in (d,g)) else None
            row['devastator_training_speedup_vs_galvatron']=g['mean_iteration_s']/d['mean_iteration_s'] if complete else None
            row['galvatron_over_devastator_search_time']=g['search_e2e_seconds']/d['search_e2e_seconds'] if complete and all(r.get('search_e2e_seconds',0)>0 for r in (d,g)) else None
            row['galvatron_over_devastator_cumulative_search_time']=g['search_cumulative_e2e_seconds']/d['search_cumulative_e2e_seconds'] if complete and all((r.get('search_cumulative_e2e_seconds') or 0)>0 for r in (d,g)) else None
            row['comparison_note']='Full-space KV changes alter model structure; throughput does not establish equal model quality.' if space=='full' else 'Common-space fixed native KV and equal candidate set; consult completion audit.'
            pairs.append(row)
    save(root/'paired_results.json',pairs)
    with (root/'paired_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    return pairs


if __name__=='__main__':
    rows=export_pairs()
    print(dict(pairs=len(rows),completed_pairs=sum(r['both_training_completed'] for r in rows)))
