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

    def test_python_nonliteral_expressions_are_not_credentials(self):
        source = '''token = tokenizer.encode(text)
token = request["token"]
token = model.token
token = current_token + offset
f(token=some_tensor, access_token=os.environ.get("ACCESS_TOKEN"))
payload = {"token": request.token}
'''
        self.assertEqual(collector.secret_findings(source, 'Megatron-LM/model.py'), [])

    def test_python_literals_still_block(self):
        for source in ('token = "sensitive_demo_12345"',
                       'token: str = "sensitive_demo_12345"',
                       'client(token="sensitive_demo_12345")',
                       'config = {"password": "sensitive_demo_12345"}',
                       'token = "sensitive_" + "demo_12345"',
                       'token = f"sensitive_demo_12345{suffix}"'):
            findings = collector.secret_findings(source, 'Megatron-LM/local.py')
            self.assertTrue(findings, source)
            self.assertNotIn('sensitive_demo_12345', json.dumps(findings))

    def test_url_variable_reference_is_not_literal_password(self):
        for value in ('$CI_JOB_TOKEN', '${CI_JOB_TOKEN}', '${H20_URL_PASSWORD}'):
            text = 'url = "https://gitlab-ci-token:' + value + '@example.invalid/repo"'
            self.assertEqual(collector.secret_findings(text, 'Megatron-LM/ci.yml'), [])
        for value in ('literal_demo_password', '${PASSWORD:-literal_default}', 'prefix${PASSWORD}'):
            self.assertTrue(collector.secret_findings('https://user:' + value + '@example.invalid', 'local.sh'))

    def test_url_redaction_preserves_original_and_records_export(self):
        source = '#!/bin/bash\nexport HTTPS_PROXY="https://demo_user:literal_demo_password@example.invalid:80"\n'
        path = self.meg/'run.sh'
        path.write_text(source, encoding='utf-8')
        dest = self.run_collect(redact_url_files=['Megatron-LM/run.sh'])
        self.assertEqual(path.read_text(encoding='utf-8'), source)
        exported = (dest/'source_payload/Megatron-LM/run.sh').read_text(encoding='utf-8')
        self.assertTrue(exported.startswith('#!/bin/bash\n'))
        self.assertIn('${H20_URL_USER}:${H20_URL_PASSWORD}', exported)
        self.assertNotIn('literal_demo_password', exported)
        self.assertNotIn('demo_user', exported)
        row = next(r for r in json.loads((dest/'source_manifest.json').read_text()) if r['path'].endswith('/run.sh'))
        self.assertTrue(row['transformed'])
        self.assertNotEqual(row['sha256'], row['original_sha256'])
        scan = json.loads((dest/'scan_report.json').read_text())
        self.assertEqual(scan['export_transformations'][0]['url_count'], 1)
        self.assertNotIn('literal_demo_password', json.dumps(scan))

    def test_no_automatic_redaction_or_bypass_of_other_secrets(self):
        path = self.meg/'run.sh'
        path.write_text('export HTTPS_PROXY="https://demo:literal_demo_password@example.invalid"\n')
        with self.assertRaises(collector.CollectionError):
            self.run_collect()
        path.write_text(path.read_text() + 'export PASSWORD="another_literal_demo_password"\n')
        with self.assertRaises(collector.CollectionError):
            self.run_collect(redact_url_files=['Megatron-LM/run.sh'])

    def test_redaction_target_must_be_present_and_allowlisted(self):
        for target in ('Megatron-LM/missing.sh', 'Megatron-LM/run.sh', '../../outside.sh'):
            with self.assertRaises(collector.CollectionError):
                self.run_collect(redact_url_files=[target])

    def test_unparseable_python_not_silently_approved(self):
        findings = collector.secret_findings('token = "long_literal_demo"\nif broken syntax\n', 'bad.py')
        self.assertTrue(findings)
        self.assertEqual(findings[0]['rule'], 'python_literal_scan_unparsed')

    def test_local_transfer_retains_findings_without_public_approval(self):
        (self.meg/'local.py').write_text('token = "sensitive_demo_12345"\n')
        dest = self.run_collect(local_transfer=True)
        report = json.loads((dest/'scan_report.json').read_text())
        self.assertEqual(report['status'], 'blocked')
        self.assertTrue(report['suspected_secrets'])
        self.assertFalse(report['public_upload_allowed'])
        self.assertNotIn('sensitive_demo_12345', json.dumps(report))
        ready = json.loads((dest/'READY.json').read_text())
        self.assertEqual(ready['status'], 'collected_for_private_review')
        self.assertFalse(ready['public_upload_allowed'])
        self.assertTrue((dest/'PRIVATE_REVIEW_REQUIRED.json').exists())
        self.assertTrue((dest/'source_payload/Megatron-LM/local.py').exists())

    def test_local_transfer_still_blocks_external_input(self):
        self.cases.write_text(json.dumps({'cases': [{'id': 'bad',
            'data_path': '../outside/train', 'tokenizer_path': 'model_from_hf/test'}]}))
        with self.assertRaises(collector.CollectionError):
            self.run_collect(local_transfer=True)


if __name__ == '__main__':
    unittest.main()
