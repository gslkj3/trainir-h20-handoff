"""Launch bounded native-loop eight-GPU checks on verified historical inputs."""
import argparse
import hashlib
import json
import os
import math
import re
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--system', choices=['galvatron', 'megatron'], required=True)
p.add_argument('--case', choices=['llama-tp4-fp16', 'llama-tp2-fp16', 'llama3-tp4-bf16', 'qwen3-tp2-bf16'], required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--runtime', type=Path, default=Path('/opt/hbv/trainir-h20-runtime'))
a = p.parse_args()
out = a.out.resolve()
out.mkdir(parents=True, exist_ok=False)
family, tpstr, precision = a.case.split('-')
tp = int(tpstr[2:])
dp = 8//(2*tp)
tokenizer, vocab, rope = {'llama':('llama2-hf',32000,10000),
                         'llama3':('llama3-hf',128256,500000),
                         'qwen3':('qwen3-hf',151936,1000000)}[family]
root = a.runtime.resolve()
meg = root/'Megatron-LM'
repo = root/('Hetu-Galvatron-dtsir' if a.system == 'galvatron' else 'Megatron-LM')
venv = Path('/opt/hbv/venv-'+a.system+'-h20')
python = str(venv/'bin/python')
here = Path(__file__).resolve().parent
env = dict(os.environ)
for key in list(env):
    if key.startswith('DTSIR_') or key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
        env.pop(key)
env.update(PATH=str(venv/'bin')+':/opt/hbv/venv-galvatron-h20/bin:/usr/local/cuda/bin:'+env['PATH'],
           PYTHONPATH=str(repo), CUDA_HOME='/usr/local/cuda', OMP_NUM_THREADS='1',
           CUDA_DEVICE_MAX_CONNECTIONS='1', AUTOMM='0', DTSIR_EXPERIMENT='off',
           H20_VALIDATION_OUT=str(out), MEGATRON_ROOT=str(meg),
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'), TRITON_CACHE_DIR=str(out/'triton'),
           TORCH_EXTENSIONS_DIR=str(out/'extensions'))
if a.system == 'megatron':
    env['CUDNN_HOME'] = '/opt/hbv/venv-galvatron-h20/lib/python3.10/site-packages/nvidia/cudnn'
    env['LD_LIBRARY_PATH'] = env['CUDNN_HOME']+'/lib:'+env.get('LD_LIBRARY_PATH', '')
cmd = [python,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=8']
if a.system == 'galvatron':
    template = Path('/opt/hbv/galvatron-h20-results/smoke-configs-v3')
    config = json.loads((template/f'tp{tp}-pp2-dp{dp}-{precision}.yaml').read_text())
    model = json.loads((template/'model.yaml').read_text())
    model.update(vocab_size=vocab, padded_vocab_size=vocab, rotary_base=rope,
                 qk_layernorm=family == 'qwen3')
    (out/'model.yaml').write_text(json.dumps(model,indent=2))
    r = config['runtime']
    # Some non-null schema defaults override model-file fields upstream.
    # Put the required semantics inline and validate the resolved configuration.
    r['model'].update(model)
    r['model']['model_config_path'] = str(out/'model.yaml')
    r['model']['padded_vocab_size'] = vocab
    r['data'].update(use_random_dataset=False, tokenizer_model=str(meg/'model_from_hf'/tokenizer),
                     data_path=[str(meg/'dataset'/family/'enwiki_text_document')],
                     data_cache_path=str(out/'dataset-cache'), split='100,0,0')
    (out/'runtime.yaml').write_text(json.dumps(config,indent=2))
    cmd += [str(here/'h20_galvatron_validation.py'),str(out/'runtime.yaml'),str(out)]
else:
    cmd += [str(here/'h20_megatron_validation.py'),
        '--use-mcore-models','--transformer-impl','transformer_engine',
        '--tensor-model-parallel-size',str(tp),'--pipeline-model-parallel-size','2',
        '--context-parallel-size','1','--sequence-parallel',
        '--num-layers','4','--hidden-size','512','--ffn-hidden-size','1376',
        '--num-attention-heads','8','--group-query-attention','--num-query-groups','4','--kv-channels','64',
        '--tokenizer-type','HuggingFaceTokenizer','--tokenizer-model',str(meg/'model_from_hf'/tokenizer),
        '--make-vocab-size-divisible-by','1','--padded-vocab-size',str(vocab),
        '--seq-length','128','--max-position-embeddings','128',
        '--micro-batch-size',str(2 if dp==1 else 1),'--global-batch-size','8','--train-iters','10',
        '--lr','1e-4','--min-lr','1e-5','--lr-decay-style','cosine','--lr-warmup-fraction','0',
        '--untie-embeddings-and-output-weights','--disable-bias-linear',
        '--attention-dropout','0','--hidden-dropout','0','--init-method-std','0.02',
        '--position-embedding-type','rope','--rotary-base',str(rope),
        '--normalization','RMSNorm','--norm-epsilon','1e-5','--swiglu','--no-persist-layer-norm',
        '--use-flash-attn','--no-masked-softmax-fusion','--attention-softmax-in-fp32',
        '--weight-decay','0.1','--clip-grad','1','--adam-beta1','0.9','--adam-beta2','0.95',
        '--no-gradient-accumulation-fusion','--'+precision,'--seed','42',
        '--data-path',str(meg/'dataset'/family/'enwiki_text_document'),
        '--data-cache-path',str(out/'dataset-cache'),'--split','100,0,0',
        '--log-interval','1','--eval-iters','0','--eval-interval','1000',
        '--no-load-optim','--no-load-rng','--distributed-backend','nccl',
        '--distributed-timeout-minutes','5','--num-workers','0']
    if precision == 'fp16': cmd += ['--loss-scale','128']
    if family == 'qwen3': cmd += ['--qk-layernorm']
(out/'launch.json').write_text(json.dumps(dict(command=cmd, system=a.system,
    data_family=family,tp=tp,pp=2,dp=dp,precision=precision,source=str(repo),
    environment={k:env[k] for k in ['CUDA_DEVICE_MAX_CONNECTIONS','OMP_NUM_THREADS','PYTHONPATH']},
    scope='Small-model real-input integration validation, not full-workload performance.'),indent=2))
with (out/'pip-freeze.txt').open('w') as f:
    subprocess.run([python,'-m','pip','freeze'],stdout=f,check=True)
with (out/'train.log').open('w') as f:
    result = subprocess.run(['timeout','--signal=TERM','--kill-after=30s','900s']+cmd,
                            env=env,cwd=repo,stdout=f,stderr=subprocess.STDOUT)
ranks = [json.loads(x.read_text()) for x in out.glob('rank[0-7].json')]
passed = (result.returncode == 0 and len(ranks)==8 and all(r['passed'] for r in ranks)
          and {r['rank'] for r in ranks} == set(range(8))
          and {r['device'] for r in ranks} == set(range(8))
          and all(len(r['updates']) == 10 for r in ranks))
effective_models = [r['effective']['model'] if a.system == 'galvatron' else r['effective'] for r in ranks]
passed = passed and all(
    m['rotary_base'] == rope and m['qk_layernorm'] == (family == 'qwen3')
    and m['padded_vocab_size'] == vocab and m['normalization'] == 'RMSNorm'
    and m['num_layers'] == 4 and m['num_attention_heads'] == 8
    and m['num_query_groups'] == 4 and m['kv_channels'] == 64 for m in effective_models)
log = (out/'train.log').read_text(errors='replace')
losses = []
if a.system == 'megatron':
    losses = [float(x) for x in re.findall(r'lm loss:\s*([^ |]+)',log)]
    skipped = re.findall(r'number of skipped iterations:\s*(\d+)',log)
    passed = passed and len(losses)==10 and all(math.isfinite(x) for x in losses)
    passed = passed and len(skipped)==10 and all(int(x)==0 for x in skipped)
status = dict(passed=passed,exit_code=result.returncode,completed_ranks=len(ranks),system=a.system,case=a.case,logged_losses=losses)
(out/'status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status),flush=True)
sys.exit(0 if passed else 1)
