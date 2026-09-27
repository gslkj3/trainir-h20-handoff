"""CPU-only collection tests with synthetic files, not experimental evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/collect_h20_sources.py'
spec = importlib.util.spec_from_file_location('collect_sources', SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.meg = self.root/'meg'
        self.gal = self.root/'gal'
        self.out = self.root/'out'
        self.out.mkdir()
        for label, root in [('Megatron-LM', self.meg), ('Hetu-Galvatron-dtsir', self.gal)]:
            for name in collector.REQUIRED[label]:
                p = root/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('# synthetic source\n', encoding='utf-8')
        for name in ('dataset/test/train.bin', 'dataset/test/train.idx'):
            p = self.meg/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'synthetic test input')
        p = self.meg/'model_from_hf/test/tokenizer.json'
        p.parent.mkdir(parents=True)
        p.write_text('{}', encoding='utf-8')
        self.cases = self.root/'cases.json'
        self.cases.write_text(json.dumps({'cases': [{'id': 'synthetic',
            'data_path': 'dataset/test/train', 'tokenizer_path': 'model_from_hf/test'}]}))

    def tearDown(self):
        self.tmp.cleanup()

    def run_collect(self, **kwargs):
        return collector.collect(self.meg, self.gal, self.out, self.cases, **kwargs)

    def test_fresh_directories_and_default_private_inventory(self):
        first, second = self.run_collect(), self.run_collect()
        self.assertNotEqual(first, second)
        self.assertTrue((first/'READY.json').is_file())
        self.assertFalse((first/'private_inputs').exists())
        rows = json.loads((first/'input_manifest.json').read_text())
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(not r['included'] and len(r['sha256']) == 64 for r in rows))

    def test_private_inputs_are_separate_and_profile_cache_excluded(self):
        cache = self.meg/'mm_logs/calc_data/data.json'
        cache.parent.mkdir(parents=True)
        cache.write_text('{"not_reusable": true}')
        dest = self.run_collect(with_inputs=True, with_data=True)
        self.assertTrue((dest/'private_inputs/Megatron-LM/dataset/test/train.bin').exists())
        self.assertFalse((dest/'source_payload/Megatron-LM/mm_logs').exists())
        self.assertFalse((dest/'source_payload/Megatron-LM/model_from_hf').exists())

    def test_missing_input_is_not_silently_ignored(self):
        (self.meg/'dataset/test/train.idx').unlink()
        with self.assertRaises(collector.CollectionError):
            self.run_collect()
        failed = next(self.out.iterdir())
        self.assertTrue((failed/'FAILED.json').exists())
        self.assertFalse((failed/'READY.json').exists())

    def test_missing_core_source_fails(self):
        (self.meg/'pretrain_gpt.py').unlink()
        with self.assertRaises(collector.CollectionError):
            self.run_collect()

    def test_secret_findings_do_not_disclose_secret(self):
        fake = 'ghp_' + 'A' * 36
        (self.meg/'local.py').write_text('token = "' + fake + '"\n')
        with self.assertRaises(collector.CollectionError):
            self.run_collect()
        failed = next(self.out.iterdir())
        report = (failed/'scan_report.json').read_text()
        self.assertNotIn(fake, report)
        self.assertIn('github_token', report)
        self.assertFalse((failed/'source_payload').exists())

    def test_symlink_escape_is_blocked(self):
        external = self.root/'outside.py'
        external.write_text('# outside source\n')
        link = self.meg/'escape.py'
        try:
            link.symlink_to(external)
        except OSError as exc:
            self.skipTest('OS does not permit symlink creation: ' + str(exc))
        with self.assertRaises(collector.CollectionError):
            self.run_collect()

    def test_external_input_path_is_blocked_even_when_file_exists(self):
        external = self.root/'external'
        external.mkdir()
        (external/'tokenizer.json').write_text('{}')
        self.cases.write_text(json.dumps({'cases': [{'id': 'synthetic',
            'data_path': 'dataset/test/train', 'tokenizer_path': str(external)}]}))
        with self.assertRaises(collector.CollectionError):
            self.run_collect()

    def test_real_eight_case_manifest_schema(self):
        # Use the actual checked-in manifest, but only synthetic input bytes.
        manifest = SCRIPT.parents[1]/'config/cases8.json'
        self.assertTrue(manifest.is_file())
        rows = collector.case_rows(manifest)
        self.assertEqual(len(rows), 8)
        for row in rows:
            inputs = row['inputs']
            for suffix in ('.bin', '.idx'):
                path = self.meg/(inputs['data_prefix'] + suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'synthetic benchmark input')
            for key in ('megatron_tokenizer', 'galvatron_tokenizer'):
                path = self.meg/inputs[key]
                if path.suffix:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'synthetic tokenizer model')
                else:
                    path.mkdir(parents=True, exist_ok=True)
                    (path/'tokenizer.json').write_text('{}')
        dest = collector.collect(self.meg, self.gal, self.out, manifest)
        ready = json.loads((dest/'READY.json').read_text())
        self.assertEqual(ready['cases'], 8)
        self.assertEqual(ready['dataset_files'], 6)
        self.assertEqual(ready['tokenizer_files'], 4)
        self.assertFalse((dest/'private_inputs').exists())

    def test_both_backend_tokenizer_paths_are_required(self):
        self.cases.write_text(json.dumps({'cases': [{'id': 'synthetic', 'inputs': {
            'data_prefix': 'dataset/test/train',
            'megatron_tokenizer': 'model_from_hf/test',
            'galvatron_tokenizer': 'model_from_hf/missing'}}]}))
        with self.assertRaises(collector.CollectionError):
            self.run_collect()

    def test_oversized_source_fails_and_is_reported(self):
        original = collector.MAX_SOURCE_BYTES
        collector.MAX_SOURCE_BYTES = 100
        try:
            (self.meg/'large.py').write_text('#' * 101)
            with self.assertRaises(collector.CollectionError):
                self.run_collect()
        finally:
            collector.MAX_SOURCE_BYTES = original
        failed = next(self.out.iterdir())
        report = json.loads((failed/'scan_report.json').read_text())
        self.assertEqual(report['oversized_source_files'][0]['path'], 'Megatron-LM/large.py')

    def test_output_inside_source_is_rejected(self):
        with self.assertRaises(collector.CollectionError):
            collector.collect(self.meg, self.gal, self.meg, self.cases)


if __name__ == '__main__':
    unittest.main()
