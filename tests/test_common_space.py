"""Standard-library-only checks. No CUDA, network or external backend imports."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('common_space', ROOT / 'scripts' / 'common_space.py')
COMMON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)


class CommonSpaceTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'config' / 'cases8.json').read_text(encoding='utf-8'))

    def test_counts_and_fixed_signatures(self):
        COMMON.validate(self.config)
        signatures = {
            40: '25b1066b1a7f70c31c591716431d4a53b817dc07ce84641f22878fb7519b481f',
            24: 'de68b0de0b52e0eb41eda41bc0fdf92b29e1737d9a1d7b7a14b95010dfd5c864',
            32: '6079cdbf54d01a136b23f33ba20e6f88f2247e8105502a2cba743cd8288ebf45',
        }
        count = 0
        for case in self.config['cases']:
            rows = COMMON.candidates(case)
            self.assertEqual(len(rows), case['expected_structural_candidates'])
            self.assertEqual(COMMON.candidate_signature(rows), signatures[len(rows)])
            self.assertEqual(COMMON.candidate_signature(rows), COMMON.candidate_signature(list(reversed(rows))))
            count += len(rows)
        self.assertEqual(count, 296)

    def test_every_row_is_structurally_valid(self):
        for case in self.config['cases']:
            for dp, pp, tp, mbs in COMMON.candidates(case):
                self.assertEqual(dp * pp * tp, 8)
                self.assertEqual(case['num_layers'] % pp, 0)
                self.assertEqual(case['native_kv_heads'] % tp, 0)
                self.assertEqual(case['num_attention_heads'] % tp, 0)
                self.assertEqual(case['global_batch_size'] % (dp * mbs), 0)
                self.assertGreaterEqual(case['global_batch_size'] // (dp * mbs), pp)

    def test_no_feasibility_or_runtime_claim(self):
        payload = COMMON.build_payload(self.config, self.config['cases'][0])
        self.assertFalse(payload['actual_model_equality_verified'])
        self.assertFalse(payload['actual_candidate_consumption_verified'])
        self.assertIsNone(payload['memory_limit_gib'])
        for row in payload['candidates']:
            self.assertIsNone(row['modeled_memory_feasible'])
            self.assertFalse(row['backend_support_verified'])
            self.assertEqual(row['training_status'], 'not_run')
            self.assertEqual(row['sp_degree'], row['tp'])
            self.assertEqual(row['runtime_sequence_parallel'], row['tp'] > 1)

    def test_model_hash_changes_for_math_not_labels(self):
        case = copy.deepcopy(self.config['cases'][0])
        old = COMMON.digest(COMMON.declared_model_spec(self.config, case))
        case['label'] = 'Display name only'
        self.assertEqual(COMMON.digest(COMMON.declared_model_spec(self.config, case)), old)
        case['norm_epsilon'] *= 10
        self.assertNotEqual(COMMON.digest(COMMON.declared_model_spec(self.config, case)), old)

    def test_qwen_audit_issues_are_preserved(self):
        cases = {c['id']: c for c in self.config['cases']}
        self.assertIn('reconciliation', cases['qwen2_1p5b_32k']['model_contract_status'])
        self.assertIn('reconciliation', cases['qwen2_7b_32k']['model_contract_status'])
        self.assertEqual(cases['qwen2_7b_32k']['inputs']['megatron_tokenizer'], 'model_from_hf/qwen3-hf')
        self.assertTrue(cases['qwen3_14b_4k']['qk_layernorm'])

    def test_candidate_equality_detects_duplicates_and_missing_rows(self):
        rows = COMMON.candidates(self.config['cases'][0])
        COMMON.require_equal(list(reversed(rows)), rows, 'test')
        with self.assertRaises(ValueError):
            COMMON.require_equal(rows + [rows[0]], rows, 'test')
        with self.assertRaises(ValueError):
            COMMON.require_equal(rows[1:], rows, 'test')

    def test_wrong_world_and_nonpositive_dimensions_rejected(self):
        case = copy.deepcopy(self.config['cases'][0])
        with self.assertRaises(ValueError):
            COMMON.candidates(case, world=16)
        case['num_layers'] = 0
        with self.assertRaises(ValueError):
            COMMON.candidates(case)

    def test_changed_protocol_rejected(self):
        self.config['measurement_protocol']['independent_runs'] = 3
        with self.assertRaises(ValueError):
            COMMON.validate(self.config)

    def test_generate_and_refuse_any_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'fresh'
            result = COMMON.generate(ROOT / 'config' / 'cases8.json', output)
            self.assertEqual(result['total_structural_candidates'], 296)
            self.assertEqual(len(list(output.glob('*.json'))), 9)
            original = (output / 'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):
                COMMON.generate(ROOT / 'config' / 'cases8.json', output)
            self.assertEqual(original, (output / 'manifest.json').read_bytes())
            empty = Path(tmp) / 'empty'
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                COMMON.generate(ROOT / 'config' / 'cases8.json', empty)


if __name__ == '__main__':
    unittest.main()
