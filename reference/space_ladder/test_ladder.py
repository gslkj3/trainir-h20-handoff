import json
from pathlib import Path
import tempfile
import unittest
from space_entry import strategy_grid, canonical
from run_ladder import check_nested, job

class Tests(unittest.TestCase):
    def test_strategy_nesting(self):
        last=set()
        for level,expected in enumerate([1,2,4,12]):
            rows=list(strategy_grid(level,32,32,[1,2]))
            now={json.dumps(x,sort_keys=True) for x in rows}
            self.assertEqual(len(now),expected)
            self.assertTrue(last<=now)
            self.assertTrue(all(x['Hybrid_MHA_MQA']==[False,32] for x in rows))
            last=now

    def test_fixed_gqa(self):
        self.assertTrue(all(x['Hybrid_MHA_MQA']==[True,8] for x in strategy_grid(3,8,32,[1,2])))

    def test_nesting_rejects_missing(self):
        a={'parallel':[4,1,1,1,1,1,1,2,32],'strategy':{},'predicted_cost':1.0}
        previous={'screened':[a],'evaluated':[a],'best':a}
        current={'screened':[],'evaluated':[],'best':None}
        with self.assertRaises(RuntimeError): check_nested(previous,current)

    def test_completed_job_never_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); original=json.dumps({'ok':True,'outcome':'completed'})
            (p/'status.json').write_text(original)
            self.assertTrue(job(p,{},Path('nonexistent.sh'),1)['ok'])
            self.assertEqual((p/'status.json').read_text(),original)

    def test_expected_oom_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'status.json').write_text(json.dumps({'ok':False,'outcome':'cuda_oom'}))
            self.assertEqual(job(p,{},Path('nonexistent.sh'),1)['outcome'],'cuda_oom')

    def test_other_failure_not_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'status.json').write_text(json.dumps({'ok':False,'outcome':'failed'}))
            with self.assertRaises(RuntimeError): job(p,{},Path('nonexistent.sh'),1)

if __name__=='__main__': unittest.main()
