"""Check paired 16-GPU results without assigning a speedup to failed runs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dtsir_common16.common5_space import require_equal, signature


NAMES = ('llama7b_2k','llama2_7b_4k','llama2_13b_4k',
         'llama3_8b_8k','qwen3_14b_4k')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('tag',help='PAIR16_TAG printed at submission')
    p.add_argument('--megatron-root',default='/data/run01/LEGACY_USER/wjy/Megatron-LM')
    p.add_argument('--galvatron-root',default='/data/run01/LEGACY_USER/wjy/dependencies/Hetu-Galvatron-dtsir')
    a=p.parse_args()
    meg=Path(a.megatron_root)/'mm_logs'/(a.tag+'_megatron')
    gal=Path(a.galvatron_root)/'mm_logs'/(a.tag+'_galvatron')
    print('case\tspace\tMeg search-stage wall s\tGal search+compute/memory Profile wall s\t'
          'Meg samples/s\tGal samples/s\tGal status')
    for name in NAMES:
        ms=read(meg/name/'search/search.json')
        mm=read(meg/name/'summary.json')
        gs=read(gal/name/'summary.json')
        gc=read(gal/name/'common_space.json')
        ge=read(gal/name/'candidate_evaluations.json')
        status=read(gal/name/'status.json')
        if ms and gc:
            if ms['common_candidate_sha256']!=gc['sha256'] or \
               ms['common_candidate_count']!=gc['count']:
                raise RuntimeError(f'{name}: candidate sets differ')
            space=f"{gc['count']} CONTRACT MATCH; evaluation incomplete"
            expected=gc['candidates']
            if signature(expected)!=gc['sha256'] or len(expected)!=gc['count']:
                raise RuntimeError(f'{name}: inconsistent common-space manifest')
            screened=[[p[0],p[1],p[4],p[7]] for p in ms['screened']]
            processed=[[r['parallel'][0],r['parallel'][1],r['parallel'][4],r['parallel'][7]]
                       for r in ms['evaluated']+ms['rejected']]
            require_equal(screened,expected,f'{name}: Megatron screened')
            require_equal(processed,expected,f'{name}: Megatron evaluated/rejected')
            if ge and ge['status']=='completed':
                require_equal([r['candidate'] for r in ge['records']],expected,
                              f'{name}: Galvatron evaluated')
                if any('error' in r for r in ge['records']):
                    raise RuntimeError(f'{name}: Galvatron completed with candidate errors')
                space=f"{gc['count']} EVALUATED MATCH"
        else:
            space='incomplete'
        fields=(name,space,
                mm.get('search_stage_wall_s') if mm else None,
                gs.get('profile_and_search_s') if gs else None,
                mm.get('samples_per_second') if mm else None,
                gs.get('training',{}).get('samples_per_second') if gs else None,
                status.get('status') if status else 'missing')
        print('\t'.join(str(x) if x is not None else 'NA' for x in fields))
    print('Wall clocks include cold operator/model profiling and process startup; '
          'Galvatron includes profile-data processing. Hardware communication calibration '
          'and selected training are excluded. Megatron evaluator-internal search_seconds '
          'is retained separately in summary.json. Failed workflows have no speedup.')


if __name__=='__main__':main()
