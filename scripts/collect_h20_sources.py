#!/usr/bin/env python3
"""Collect reviewed source and input inventories, without modifying either repo.

Nothing is uploaded. A fresh output directory is always allocated. READY.json is
written only after all checks pass; private_inputs must not be committed to Git.
Passing the scan does not authorize public redistribution of source or inputs.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


MAX_SOURCE_BYTES = 5 * 1024 * 1024
SKIP_DIRS = {
    '.git', '.hg', '.svn', '__pycache__', 'build', 'dist', '.cache',
    '.pytest_cache', 'mm_logs', 'logs', 'log', 'checkpoints', 'wandb',
    'tensorboard', 'dataset', 'datasets_cache', 'model_from_hf',
    '.venv', 'venv', 'envs', 'conda', 'outputs', 'results', 'private_inputs',
    'calc_data', 'comm_data', 'triton_cache', 'torchinductor_cache',
}
EXTENSIONS = {
    '.py', '.pyi', '.sh', '.sbatch', '.json', '.yaml', '.yml', '.toml',
    '.cfg', '.ini', '.txt', '.md', '.rst', '.cpp', '.cc', '.c', '.cu',
    '.cuh', '.h', '.hpp', '.cmake', '.in', '.patch', '.diff',
}
SOURCE_NAMES = {'Makefile', 'LICENSE', 'NOTICE', 'MANIFEST.in', '.gitmodules', 'CMakeLists.txt'}
PRIVATE_NAMES = {'.env', '.netrc', '.npmrc', '.pypirc', 'credentials', 'id_rsa', 'id_ed25519'}
EXTRA_DIRS = {
    'Megatron-LM': ('megatron', 'tools', 'requirements', 'Test_Auto',
                    'dtsir_common16', 'paired16', 'dtsir_ir_tests',
                    'dtsir_space_tests', 'dtsir_common32', 'paired32'),
    'Hetu-Galvatron-dtsir': ('galvatron', 'tools', 'requirements', 'configs',
                            'config', 'dtsir_galvatron6', 'dtsir_galvatron32'),
}
REQUIRED = {
    'Megatron-LM': ('pretrain_gpt.py', 'test_parallel_model.py', 'dtsir_collect.py',
                    'megatron/core/__init__.py', 'megatron/training/initialize.py'),
    'Hetu-Galvatron-dtsir': ('setup.py', 'galvatron/__init__.py',
                            'galvatron/models/gpt/train_dist.py',
                            'galvatron/core/search_engine/search_engine.py'),
}
INPUT_EXTENSIONS = {'.json', '.txt', '.model', '.tiktoken', '.jinja', '.jinja2', '.py', '.vocab', '.merges'}
# Values are never included in a finding. These checks are deliberately bounded:
# passing this scan is not a guarantee that arbitrary proprietary material is safe.
SECRET_RULES = (
    ('private_key', re.compile(r'-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----')),
    ('github_token', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b')),
    ('aws_access_key', re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
    ('credential_in_url', re.compile(r'https?://[^\s/:@]+:[^\s/@]+@')),
    ('literal_secret', re.compile(
        r'''(?im)^\s*(?:export\s+)?["']?(?:[A-Z0-9_]*(?:TOKEN|PASSWORD|SECRET|API_KEY|ACCESS_KEY))["']?\s*[:=]\s*["']?([A-Za-z0-9_./+!@=-]{12,})''')),
)
URL_USERINFO = re.compile(r'(https?://)([^\s/:@]+):([^\s/@]+)@')
ENV_REFERENCE = re.compile(r'(?:\$[A-Za-z_][A-Za-z_0-9]*|\$\{[A-Za-z_][A-Za-z_0-9]*\}|\$\{\{\s*[^{}\r\n]+\s*\}\})')
SECRET_NAME = re.compile(r'(?:.*_)?(?:token|password|secret|api_key|access_key)$', re.I)


def python_literal_findings(content, relative):
    """Inspect literal values, not the spelling of Python expressions."""
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        # Keep conservative behavior for syntax this Python cannot parse.
        pattern = dict(SECRET_RULES)['literal_secret']
        return [{'path': relative, 'line': content.count('\n', 0, m.start()) + 1,
                 'rule': 'python_literal_scan_unparsed'} for m in pattern.finditer(content)]

    def target_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            return node.slice.value if isinstance(node.slice.value, str) else ''
        return ''

    def literal_parts(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            return [node.value]
        if isinstance(node, (ast.BinOp, ast.JoinedStr)):
            return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant)
                    and isinstance(n.value, (str, bytes))]
        return []

    pairs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            pairs.extend((target_name(t), node.value, node.lineno) for t in node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            pairs.append((target_name(node.target), node.value, node.lineno))
        elif isinstance(node, ast.keyword):
            pairs.append((node.arg or '', node.value, node.lineno))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    pairs.append((key.value, value, key.lineno))
    findings = []
    for name, value, line in pairs:
        if not SECRET_NAME.fullmatch(name):
            continue
        parts = literal_parts(value)
        # Include split/f-string literals without treating tensor arithmetic as credentials.
        if sum(len(p) for p in parts) >= 12:
            findings.append({'path': relative, 'line': line, 'rule': 'literal_secret'})
    return [dict(t) for t in sorted({tuple(sorted(row.items())) for row in findings})]


def redact_url_userinfo(content):
    # Replacements are confined to explicit caller-selected exported files.
    return URL_USERINFO.subn(lambda m: m.group(1) + '${H20_URL_USER}:${H20_URL_PASSWORD}@', content)


class CollectionError(RuntimeError):
    pass


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked(path, root):
    """Resolve before reading, including symlinks in ancestor directories."""
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (ValueError, OSError, RuntimeError):
        raise CollectionError('Missing input or path escapes repository: ' + str(path)) from None
    return resolved


def eligible(relative):
    return not any(p in SKIP_DIRS or p.endswith('.egg-info') for p in relative.parts) and (
        relative.suffix.lower() in EXTENSIONS or relative.name in SOURCE_NAMES
    ) and relative.name not in PRIVATE_NAMES and not relative.name.startswith('.env.')


def git_capture(root, *args):
    try:
        proc = subprocess.run(['git', '-C', str(root), *args], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def provenance(root):
    top = git_capture(root, 'rev-parse', '--show-toplevel')
    if not top or Path(os.fsdecode(top).strip()).resolve() != root:
        return {'git_available': False, 'head': None, 'dirty': None,
                'note': 'Source snapshot is not a repository root; hashes identify this snapshot.'}
    head = git_capture(root, 'rev-parse', 'HEAD')
    status = git_capture(root, 'status', '--porcelain=v1', '-z', '--untracked-files=all')
    # Do not publish diff contents, environment, remote URLs, or untracked filenames.
    return {'git_available': True, 'head': head.decode().strip() if head else None,
            'dirty': bool(status) if status is not None else None}


def walk_safe(folder, root):
    stack = [(folder, frozenset())]
    while stack:
        path, ancestors = stack.pop()
        resolved = checked(path, root)
        if resolved.is_dir():
            if resolved in ancestors:
                raise CollectionError('Directory symlink cycle: ' + str(path))
            for child in sorted(path.iterdir(), reverse=True):
                if child.name in SKIP_DIRS or child.name.endswith('.egg-info'):
                    continue
                stack.append((child, ancestors | {resolved}))
        elif resolved.is_file():
            yield path


def source_files(root, label, git_info):
    selected = set()
    if git_info['git_available']:
        tracked = git_capture(root, 'ls-files', '-z')
        if tracked is None:
            raise CollectionError('Cannot enumerate tracked source files: ' + str(root))
        for item in tracked.split(b'\0'):
            if item:
                p = root / os.fsdecode(item)
                if eligible(p.relative_to(root)):
                    # A tracked deletion is part of the working snapshot, not a
                    # request to resurrect the deleted upstream version.
                    if not p.exists() and not p.is_symlink():
                        continue
                    checked(p, root)
                    if p.is_file():
                        selected.add(p)
    for p in root.iterdir():
        if eligible(p.relative_to(root)) and (p.is_file() or p.is_symlink()):
            checked(p, root)
            if p.is_file():
                selected.add(p)
    for name in EXTRA_DIRS[label]:
        folder = root / name
        if folder.exists() or folder.is_symlink():
            for p in walk_safe(folder, root):
                if eligible(p.relative_to(root)):
                    selected.add(p)
    return sorted(selected)


def secret_findings(content, relative):
    findings = []
    for name, pattern in SECRET_RULES:
        if name == 'literal_secret' and relative.endswith('.py'):
            findings.extend(python_literal_findings(content, relative))
            continue
        for match in pattern.finditer(content):
            if name == 'credential_in_url':
                parsed = URL_USERINFO.fullmatch(match.group(0))
                if parsed and ENV_REFERENCE.fullmatch(parsed.group(3)):
                    continue  # Reference only; generic key/token patterns still run.
            if name == 'literal_secret':
                value = match.group(1).lower()
                if value.startswith(('example', 'placeholder', 'your_', 'replace_', 'dummy', 'changeme')):
                    continue
            findings.append({'path': relative, 'line': content.count('\n', 0, match.start()) + 1,
                             'rule': name})
    return findings


def case_rows(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict) and 'cases' in data:
        data = data['cases']
    rows = list(data.values()) if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise CollectionError('Case manifest is empty or invalid: ' + str(path))
    return rows


def input_files(megatron, cases):
    data_files, tokenizer_files = set(), set()
    for c in cases:
        inputs = c.get('inputs', {})
        data = c.get('data_path', c.get('data', inputs.get('data_path', inputs.get('data_prefix'))))
        legacy_tok = c.get('tokenizer_path', c.get('tokenizer', inputs.get('tokenizer_path', inputs.get('tokenizer'))))
        tokenizers = [inputs[k] for k in ('megatron_tokenizer', 'galvatron_tokenizer') if k in inputs]
        if legacy_tok is not None:
            tokenizers.append(legacy_tok)
        if not isinstance(data, str) or not tokenizers or any(not isinstance(t, str) or not t for t in tokenizers):
            raise CollectionError('Each case needs a data_path/data_prefix and explicit tokenizer path(s)')
        for suffix in ('.bin', '.idx'):
            p = megatron / (data + suffix)
            if not checked(p, megatron).is_file():
                raise CollectionError('Missing training data: ' + str(p))
            data_files.add(p)
        for tok in sorted(set(tokenizers)):
            p = megatron / tok
            resolved = checked(p, megatron)
            folder = p if resolved.is_dir() else p.parent
            found = []
            for f in walk_safe(folder, megatron):
                if f.suffix.lower() in INPUT_EXTENSIONS:
                    found.append(f)
            if not found:
                raise CollectionError('Tokenizer has no recognized input files: ' + str(p))
            tokenizer_files.update(found)
    return {'tokenizers': sorted(tokenizer_files), 'datasets': sorted(data_files)}


def collect(megatron, galvatron, out_parent, cases_path, with_inputs=False, with_data=False,
            redact_url_files=(), local_transfer=False):
    roots = {'Megatron-LM': Path(megatron).resolve(), 'Hetu-Galvatron-dtsir': Path(galvatron).resolve()}
    parent = Path(out_parent).resolve()
    requested_redactions = set(redact_url_files)
    allowed_redactions = {'Megatron-LM/run.sh', 'Megatron-LM/test1.sh'}
    if requested_redactions - allowed_redactions:
        raise CollectionError('URL redaction is only supported for the two audited auxiliary scripts')
    if not parent.is_dir():
        raise CollectionError('--out-parent must be an existing directory')
    for root in roots.values():
        try:
            parent.relative_to(root)
        except ValueError:
            continue
        raise CollectionError('Output parent must be outside both source repositories')
    dest = Path(tempfile.mkdtemp(prefix='h20_sources_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_', dir=parent))
    print('OUTPUT: ' + str(dest), flush=True)
    findings, oversized, source_manifest, inputs_manifest = [], [], [], []
    transformations, matched_redactions = [], set()
    try:
        for label, root in roots.items():
            if not root.is_dir():
                raise CollectionError('Missing source repository: ' + str(root))
            for name in REQUIRED[label]:
                if not checked(root / name, root).is_file():
                    raise CollectionError('Missing required source: ' + str(root / name))
        rows = case_rows(Path(cases_path))
        inputs = input_files(roots['Megatron-LM'], rows)
        records = {label: provenance(root) for label, root in roots.items()}
        planned = []
        for label, root in roots.items():
            for p in source_files(root, label, records[label]):
                rel = label + '/' + p.relative_to(root).as_posix()
                target = checked(p, root)
                size = target.stat().st_size
                if size > MAX_SOURCE_BYTES:
                    oversized.append({'path': rel, 'bytes': size, 'limit': MAX_SOURCE_BYTES})
                    continue
                content = target.read_bytes()
                try:
                    decoded = content.decode('utf-8-sig')
                except UnicodeDecodeError:
                    raise CollectionError('Source file is not UTF-8 text; review before publication: ' + rel) from None
                if '\0' in decoded:
                    raise CollectionError('Binary content in source file: ' + rel)
                export_content = content
                if rel in requested_redactions:
                    matched_redactions.add(rel)
                    decoded, count = redact_url_userinfo(decoded)
                    export_content = decoded.encode('utf-8')
                    transformations.append({'path': rel, 'operation': 'replace_url_userinfo_with_environment_references',
                                            'url_count': count, 'original_source_modified': False})
                findings.extend(secret_findings(decoded, rel))
                planned.append((p, root, rel, size, hashlib.sha256(content).hexdigest(), export_content))
        if requested_redactions != matched_redactions:
            raise CollectionError('An explicitly selected URL-redaction source file was not collected')
        scan = {'status': 'blocked' if findings or oversized else 'passed',
                'suspected_secrets': findings, 'oversized_source_files': oversized,
                'export_transformations': transformations,
                'transfer_mode': 'private_local_review' if local_transfer else 'strict',
                'public_upload_allowed': False,
                'scope': 'Pattern scan only; manual license/privacy review is still required.'}
        save(dest/'scan_report.json', scan)
        if oversized or (findings and not local_transfer):
            raise CollectionError('Source audit blocked collection; inspect scan_report.json (secret values omitted)')
        if local_transfer:
            save(dest/'PRIVATE_REVIEW_REQUIRED.json', {
                'public_upload_allowed': False, 'unresolved_findings': len(findings),
                'instruction': 'Private local review only. Do not upload this archive or its unreviewed source to GitHub. '
                               'Review every finding and exclude private_inputs before any public publication.'})
        for p, root, rel, size, digest, export_content in planned:
            output = dest/'source_payload'/rel
            output.parent.mkdir(parents=True, exist_ok=True)
            if sha256(checked(p, root)) != digest:
                raise CollectionError('Source changed during collection: ' + rel)
            output.write_bytes(export_content)
            exported_digest = hashlib.sha256(export_content).hexdigest()
            if sha256(output) != exported_digest:
                raise CollectionError('Export copy verification failed: ' + rel)
            if p.suffix in ('.sh', '.sbatch'):
                output.chmod(0o755)
            source_manifest.append({'path': rel, 'bytes': len(export_content), 'sha256': exported_digest,
                                    'original_bytes': size, 'original_sha256': digest,
                                    'transformed': exported_digest != digest})
        for kind, paths in inputs.items():
            include = with_inputs if kind == 'tokenizers' else with_data
            for p in paths:
                source = checked(p, roots['Megatron-LM'])
                rel = 'Megatron-LM/' + p.relative_to(roots['Megatron-LM']).as_posix()
                digest = sha256(source)
                row = {'kind': kind, 'path': rel, 'bytes': source.stat().st_size,
                       'sha256': digest, 'included': include}
                if include:
                    output = dest/'private_inputs'/rel
                    output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, output)
                    if sha256(output) != digest:
                        raise CollectionError('Input changed during collection: ' + rel)
                inputs_manifest.append(row)
        save(dest/'source_manifest.json', source_manifest)
        save(dest/'input_manifest.json', inputs_manifest)
        save(dest/'provenance.json', records)
        save(dest/'READY.json', {'status': 'collected_for_private_review' if local_transfer else 'collected',
             'public_upload_allowed': False, 'source_files': len(source_manifest),
             'cases': len(rows), 'tokenizer_files': len(inputs['tokenizers']),
             'dataset_files': len(inputs['datasets']), 'with_inputs': with_inputs,
             'with_data': with_data, 'case_manifest_sha256': sha256(Path(cases_path)),
             'scope': 'Source collection only; not installation, training, or permission to redistribute inputs.'})
    except Exception as exc:
        save(dest/'FAILED.json', {'status': 'failed', 'error': str(exc),
                                 'note': 'Do not upload a partial collection.'})
        if not (dest/'scan_report.json').exists():
            save(dest/'scan_report.json', {'status': 'incomplete', 'suspected_secrets': findings,
                                          'oversized_source_files': oversized})
        raise CollectionError(str(exc) + '\nDiagnostic directory: ' + str(dest)) from None
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--megatron', required=True)
    parser.add_argument('--galvatron', required=True)
    parser.add_argument('--out-parent', required=True)
    parser.add_argument('--cases', default=str(Path(__file__).resolve().parents[1]/'config/cases8.json'))
    parser.add_argument('--with-inputs', action='store_true', help='Copy tokenizer bytes to private_inputs (otherwise inventory only).')
    parser.add_argument('--with-data', action='store_true', help='Copy dataset .bin/.idx bytes to private_inputs (otherwise inventory only).')
    parser.add_argument('--local-transfer', action='store_true',
                        help='Explicit private local review mode: retain suspected-secret findings without blocking collection; never approves public upload.')
    parser.add_argument('--redact-url-file', action='append', default=[],
                        choices=['Megatron-LM/run.sh', 'Megatron-LM/test1.sh'],
                        help='Replace URL credentials in this exported auxiliary script only; original stays unchanged.')
    args = parser.parse_args()
    try:
        dest = collect(args.megatron, args.galvatron, args.out_parent, args.cases, args.with_inputs, args.with_data,
                       args.redact_url_file, args.local_transfer)
    except CollectionError as exc:
        print('COLLECTION FAILED: ' + str(exc), file=sys.stderr)
        return 1
    print('READY: ' + str(dest/'READY.json'))
    print('No upload performed. Never commit private_inputs to the source repository.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
