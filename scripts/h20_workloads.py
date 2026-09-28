"""Explicit eight-workload model/training contracts; shared by audit and runners."""
import copy
import json
from h20_campaign import *

def cases(root=DEFAULT_ROOT):
    return {c['id']:c for c in json.loads((root/'cases8.json').read_text())['cases']}

def model_fields(c):
    return dict(model_size='qwen2' if c['id'].startswith('qwen') else 'llama2',
        hidden_size=c['hidden_size'],ffn_hidden_size=c['ffn_hidden_size'],num_layers=c['num_layers'],
        num_attention_heads=c['num_attention_heads'],num_query_groups=c['native_kv_heads'],kv_channels=128,
        vocab_size=c['declared_vocab_size'],padded_vocab_size=c['declared_vocab_size'],
        normalization='RMSNorm',norm_epsilon=c['norm_epsilon'],layernorm_epsilon=c['norm_epsilon'],
        activation_func='torch.nn.functional.silu',gated_linear_unit=True,position_embedding_type='rope',
        rotary_base=c['rotary_base'],rotary_percent=1.0,rotary_interleaved=False,
        rotary_seq_len_interpolation_factor=None,apply_rope_fusion=False,
        add_bias_linear=False,add_qkv_bias=c['add_qkv_bias'],qk_layernorm=c['qk_layernorm'],
        untie_embeddings_and_output_weights=c['untie_embeddings_and_output_weights'],
        make_vocab_size_divisible_by=1,initialize_on_meta=1,print_loss=1,dropout_prob=0.0,
        attention_dropout=0.0,hidden_dropout=0.0)

def galv_runtime(c, out, root=DEFAULT_ROOT):
    from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
    r=GalvatronRuntimeArgs().model_dump(exclude={'model':{'params_dtype'}})
    r['model'].update(model_fields(c))
    r['parallel'].update(pp_deg=1,global_tp_deg=1,global_cp_deg=1,global_ep_deg=1,
        global_checkpoint=0,sdp=0,vocab_tp=1,vocab_sdp=0,mixed_precision=c['dtype'],
        default_dp_type='ddp',pipeline_type='pipedream_flush',use_ulysses=False,
        reduce_in_fp32=True,entropy_in_fp32=True)
    r['train'].update(train_iters=10,eval_iters=0,iteration=0,global_batch_size=c['global_batch_size'],
        micro_batch_size=1,chunks=c['global_batch_size']//8,seq_length=c['seq_length'],
        sequence_parallel=True,use_flash_attn=True,lr=1e-6,min_lr=1e-7,
        lr_decay_style='cosine',lr_warmup_fraction=.01,init_method_std=.01,weight_decay=.1,
        adam_beta1=.9,adam_beta2=.95,adam_eps=1e-8,clip_grad=1.,num_workers=0,seed=42)
    r['data'].update(use_random_dataset=False,data_path=[str(MEG/c['inputs']['data_prefix'])],
        split='100,0,0',tokenizer_type='HuggingFaceTokenizer',
        tokenizer_model=str(MEG/c['inputs']['galvatron_tokenizer']),data_cache_path=str(out/'dataset-cache'))
    r['profile'].update(profile=0,exit_after_profiling=0,save_profiled_memory=0)
    r['ckpt'].update(load=None,save=None)
    r['distributed_timeout_minutes']=15
    return r

def meg_arguments(c, out):
    # Use the verified local tokenizer bytes for both backends, not a new HF tokenizer.
    args=['--use-mcore-models','--transformer-impl','transformer_engine',
        '--tensor-model-parallel-size','1','--pipeline-model-parallel-size','1','--context-parallel-size','1',
        '--num-layers',str(c['num_layers']),'--hidden-size',str(c['hidden_size']),
        '--ffn-hidden-size',str(c['ffn_hidden_size']),'--num-attention-heads',str(c['num_attention_heads']),
        '--kv-channels','128','--group-query-attention','--num-query-groups',str(c['native_kv_heads']),
        '--tokenizer-type','HuggingFaceTokenizer','--tokenizer-model',str(MEG/c['inputs']['galvatron_tokenizer']),
        '--make-vocab-size-divisible-by','1','--padded-vocab-size',str(c['declared_vocab_size']),
        '--seq-length',str(c['seq_length']),'--max-position-embeddings',str(c['seq_length']),
        '--micro-batch-size','1','--global-batch-size',str(c['global_batch_size']),'--train-iters','10',
        '--lr','1e-6','--min-lr','1e-7','--lr-decay-style','cosine','--lr-warmup-fraction','0.01',
        '--disable-bias-linear','--attention-dropout','0','--hidden-dropout','0','--init-method-std','0.01',
        '--position-embedding-type','rope','--rotary-base',str(c['rotary_base']),
        '--normalization','RMSNorm','--norm-epsilon',str(c['norm_epsilon']),'--swiglu','--no-persist-layer-norm',
        '--use-flash-attn','--no-masked-softmax-fusion','--attention-softmax-in-fp32',
        '--weight-decay','0.1','--clip-grad','1','--adam-beta1','0.9','--adam-beta2','0.95','--adam-eps','1e-8',
        '--no-gradient-accumulation-fusion','--'+c['dtype'],'--seed','42',
        '--accumulate-allreduce-grads-in-fp32',
        '--data-path',str(MEG/c['inputs']['data_prefix']),'--data-cache-path',str(out/'dataset-cache'),
        '--split','100,0,0','--log-interval','1','--eval-iters','0','--eval-interval','100000',
        '--no-load-optim','--no-load-rng','--distributed-backend','nccl','--distributed-timeout-minutes','15',
        '--num-workers','0']
    if c['untie_embeddings_and_output_weights']: args+=['--untie-embeddings-and-output-weights']
    if c['add_qkv_bias']: args+=['--add-qkv-bias']
    if c['qk_layernorm']: args+=['--qk-layernorm']
    if c['dtype']=='fp16': args+=['--loss-scale','128']
    return args

def megatron_semantic_adapter():
    """Configure native optimizer/init through existing hooks, without changing algorithms."""
    from megatron.training import training
    native_optimizer=training.get_megatron_optimizer
    def optimizer(config, model, *args, **kwargs):
        # Positional argument 0 is no_weight_decay_cond; use the provided hook.
        args=list(args)
        if args: args[0]=lambda name,p: False
        else: kwargs['no_weight_decay_cond']=lambda name,p: False
        return native_optimizer(config,model,*args,**kwargs)
    training.get_megatron_optimizer=optimizer
    import gpt_builders
    native_config=gpt_builders.core_transformer_config_from_args
    def config(*a,**kw):
        c=native_config(*a,**kw)
        c.output_layer_init_method=c.init_method
        return c
    gpt_builders.core_transformer_config_from_args=config

def galvatron_qk_adapter():
    from galvatron.core.runtime.models import modules
    from galvatron.core.runtime.transformer.norm import GalvatronNorm
    native=modules.SelfAttention
    class QKNormAttention(native):
        def __init__(self, config, submodules, *a, **kw):
            if config.qk_layernorm:
                submodules=copy.copy(submodules)
                submodules.q_layernorm=GalvatronNorm;submodules.k_layernorm=GalvatronNorm
            super().__init__(config,submodules,*a,**kw)
    modules.SelfAttention=QKNormAttention
