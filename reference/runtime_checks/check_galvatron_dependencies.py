"""Read-only source/import audit; no installation, training or GPU allocation."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='.')
    parser.add_argument('--out', default=None)
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    if not (repo / 'galvatron/core').is_dir():
        raise SystemExit('Run from the Hetu-Galvatron repository or pass --repo.')
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = Path(args.out or f'galvatron_dependency_audit_{stamp}.json').resolve()
    if out.exists():
        raise SystemExit(f'Report already exists; not overwriting: {out}')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(repo) + os.pathsep + env.get('PYTHONPATH', '')
    checks = [
        ('declared dependencies', [sys.executable, '-m', 'pip', 'check']),
    ]
    modules = [
        ('torch', 'torch', None),
        ('search native extension', 'galvatron_dp_core', None),
        ('search formatting', 'rich.pretty', 'pretty_repr'),
        ('configuration', 'galvatron.core.arguments', 'load_with_hydra'),
        ('runtime schema', 'galvatron.core.runtime.args_schema', 'GalvatronRuntimeArgs'),
        ('profile schema', 'galvatron.core.profiler.args_schema', 'GalvatronModelProfilerArgs'),
        ('model profiling', 'galvatron.core.profiler.model_profiler', 'ModelProfiler'),
        ('search schema', 'galvatron.core.search_engine.args_schema', 'GalvatronSearchArgs'),
        ('FULL search engine', 'galvatron.core.search_engine.search_engine', 'GalvatronSearchEngine'),
        ('model adapter', 'galvatron.utils.hf_config_adapter', 'resolve_model_config'),
        ('search model layers', 'galvatron.utils.hf_config_adapter', 'model_layer_configs'),
        ('search model naming', 'galvatron.utils.hf_config_adapter', 'model_name'),
        ('tokenizer', 'galvatron.core.runtime.datasets.megatron.tokenizer', 'build_tokenizer'),
        ('Qwen runtime modules', 'galvatron.core.runtime.models', 'modules'),
        ('Qwen normalization', 'galvatron.core.runtime.transformer.norm', 'GalvatronNorm'),
        ('flash attention', 'flash_attn', None),
        ('fused rms norm', 'flash_attn.ops.rms_norm', None),
        ('fused layer norm', 'flash_attn.ops.layer_norm', None),
    ]
    for label, module, symbol in modules:
        code = f'import importlib; m=importlib.import_module({module!r}); '
        if symbol:
            code += f'getattr(m, {symbol!r}); '
        code += 'print(getattr(m,"__file__",None))'
        checks.append((label, [sys.executable, '-c', code]))
    # Import the actual training entry without running its __main__ workflow,
    # as the experiment harness itself does.
    train = repo / 'galvatron/models/gpt/train_dist.py'
    checks.append(('FULL training entry', [sys.executable, '-c',
        f'import runpy; runpy.run_path({str(train)!r}, run_name="dependency_audit"); print("OK")']))
    checks.append(('search default schema construction', [sys.executable, '-c',
        'from galvatron.core.search_engine.args_schema import GalvatronSearchArgs; '
        'a=GalvatronSearchArgs(); print(type(a).__name__)']))
    # Inspect all hardware entry points actually used by run_six.py. Import
    # their unconditional absolute imports, not the hardware programs themselves.
    source_errors = []
    hardware_imports = set()
    for name in ('profile_allreduce.py', 'profile_p2p.py', 'profile_all2all.py', 'profile_overlap.py'):
        path = repo / 'galvatron/profile_hardware' / name
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    hardware_imports.update(a.name for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    hardware_imports.add(node.module)
        except Exception as exc:
            source_errors.append(dict(path=str(path), error=repr(exc)))
    for module in sorted(hardware_imports):
        code = f'import importlib; importlib.import_module({module!r}); print("OK")'
        checks.append(('hardware import: ' + module, [sys.executable, '-c', code]))
    results = []
    for label, cmd in checks:
        print('CHECK ' + label, flush=True)
        start = time.monotonic()
        try:
            p = subprocess.run(cmd, cwd=repo, env=env, text=True,
                encoding='utf-8', errors='replace', stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=args.timeout)
            row = dict(check=label, passed=p.returncode == 0,
                       returncode=p.returncode, output=p.stdout)
        except subprocess.TimeoutExpired as exc:
            raw = exc.stdout or ''
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='replace')
            row = dict(check=label, passed=False, returncode=None,
                       output='TIMEOUT (not proof of missing dependency)\n' + raw)
        row['seconds'] = round(time.monotonic() - start, 3)
        results.append(row)
        print(('PASS ' if row['passed'] else 'FAIL ') + label, flush=True)
        if not row['passed']:
            print(row['output'][-6000:], flush=True)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    report = dict(python=sys.executable, repo=str(repo), commit=revision.stdout.strip(),
        checks=results, source_errors=source_errors,
        passed=all(r['passed'] for r in results) and not source_errors,
        scope='CPU-side dependency/import checks only. No CUDA execution, '
              'full configuration validation, model correctness or profiling-data validation.')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('REPORT ' + str(out), flush=True)
    print('IMPORT AUDIT ' + ('PASS' if report['passed'] else 'NEEDS REVIEW'), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
