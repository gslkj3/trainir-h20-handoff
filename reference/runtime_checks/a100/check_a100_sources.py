"""CPU import/provenance diagnostics only. No pip installs, training or Slurm jobs."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback

def within(path, root):
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('system', choices=('megatron','galvatron'))
    ns=parser.parse_args()
    root=Path(os.environ['A100_WORK']).resolve()
    project=root/('Megatron-LM' if ns.system=='megatron' else 'Hetu-Galvatron-dtsir')
    expected=Path.home()/'.conda/envs'/('dtsir-a100' if ns.system=='megatron' else 'galvatron-a100')
    if Path(sys.prefix).resolve()!=expected.resolve():
        raise SystemExit(f'Wrong Python environment: {sys.prefix}, expected {expected}')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    result=dict(system=ns.system,python=sys.executable,prefix=sys.prefix,
        arch=platform.machine(),project=str(project),sys_path=sys.path,
        ld_library_path=os.environ.get('LD_LIBRARY_PATH'),ld_preload=os.environ.get('LD_PRELOAD'),
        checks=[],warnings=[])
    dest=Path(tempfile.mkdtemp(prefix='a100_cpu_'+ns.system+'_',dir=root))
    errors=[]
    manifest=json.loads((root/'migration_metadata/source_manifest.json').read_text())
    for row in manifest:
        p=root/row['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            errors.append('Source manifest mismatch: '+row['path'])
    report_path=Path.home()/'a100_transfer/migration_report.json'
    if report_path.is_file():
        report=json.loads(report_path.read_text())
        result['archive_report']=report
        if report['source_files']!=len(manifest):
            result['warnings'].append(f"Report source_files={report['source_files']}, manifest={len(manifest)}; verify package batch before formal runs.")
    result['manifest_files']=len(manifest)
    result['source_revision']={}
    for repo in ('Megatron-LM','Hetu-Galvatron-dtsir'):
        p=root/'migration_metadata'/repo/'HEAD.txt'
        result['source_revision'][repo]=p.read_text().strip() if p.is_file() else 'MISSING'
    # Save the actual build configuration before deciding how to bind/rebuild extensions.
    snapshots=dest/'source_snapshot'
    for relative in ('setup.py','pyproject.toml','CMakeLists.txt','galvatron/core/runtime/datasets/megatron/Makefile'):
        p=project/relative
        if p.is_file():
            target=snapshots/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(p.read_bytes())
    modules=['torch','numpy','scipy','pandas','sklearn','flash_attn',
        'flash_attn.ops.rms_norm','flash_attn.ops.layer_norm','transformer_engine.pytorch',
        'apex','amp_C','fused_layer_norm_cuda']
    if ns.system=='megatron':
        modules+=['megatron','test_parallel_model','dtsir_collect','pretrain_gpt']
    else:
        modules+=['rich','hydra','omegaconf','pydantic','einops','transformers',
            'galvatron','galvatron_dp_core','galvatron.core.search_engine.search_engine',
            'galvatron.models.gpt.train_dist']
    package='megatron' if ns.system=='megatron' else 'galvatron'
    for name in modules:
        print('CHECK',ns.system,name,flush=True)
        row=dict(module=name)
        try:
            module=importlib.import_module(name)
            location=getattr(module,'__file__',None)
            row.update(import_ok=True,path=location,version=str(getattr(module,'__version__','')))
            if (name==package or name.startswith(package+'.') or name in ('test_parallel_model','dtsir_collect','pretrain_gpt')):
                paths=[location] if location else list(getattr(module,'__path__',[]))
                if not paths or not all(within(p,project) for p in paths):
                    raise RuntimeError(f'Wrong source binding: {paths}; expected under {project}')
            if name=='galvatron_dp_core' and (not location or not within(location,project)):
                result['warnings'].append('Search extension is not rebuilt in the migrated tree: '+str(location))
            if name=='sklearn':
                from sklearn.metrics import r2_score
                if r2_score([1,2,3],[1,2,3])!=1.0: raise RuntimeError('sklearn computation failed')
            print('OK',name,location,flush=True)
        except Exception:
            row.update(import_ok=False,error=traceback.format_exc())
            errors.append(name)
            print(row['error'],flush=True)
        result['checks'].append(row)
    try:
        check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,timeout=120)
        result['pip_check']=dict(returncode=check.returncode,stdout=check.stdout,stderr=check.stderr)
    except Exception as exc:
        result['pip_check']=dict(error=repr(exc))
    if ns.system=='galvatron':
        result['search_extension_sources']=[str(p.relative_to(project)) for p in project.rglob('*')
            if p.is_file() and p.suffix in ('.cpp','.cc','.h','.hpp') and 'build' not in p.parts]
    result['errors']=errors
    result['scope']='CPU-only source/import audit. Not a GPU training or numerical validation.'
    (dest/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({k:result[k] for k in ('python','arch','source_revision','manifest_files','warnings','errors','pip_check')},indent=2),flush=True)
    print('AUDIT DIRECTORY:',dest,flush=True)
    return 1 if errors else 0

if __name__=='__main__':
    raise SystemExit(main())
