"""Export actual parsed eight-model configurations without constructing full GPU models."""
import argparse
import json
import os
import sys
from h20_campaign import *
from h20_workloads import cases, galv_runtime, meg_arguments

p=argparse.ArgumentParser();p.add_argument('--system',choices=['galvatron','megatron'],required=True)
p.add_argument('--root',type=Path,default=DEFAULT_ROOT);a=p.parse_args()
out=a.root/'model_audit'/a.system;out.mkdir(parents=True,exist_ok=True)
for name,c in cases(a.root).items():
    folder=out/name;folder.mkdir(exist_ok=True)
    if a.system=='galvatron':
        from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
        from galvatron.utils.hf_config_adapter import resolve_model_config
        r=galv_runtime(c,folder,a.root);args=GalvatronRuntimeArgs.model_validate(r)
        resolve_model_config(args)
        save(folder/'runtime.yaml',{'runtime':r});save(folder/'effective.json',args.model_dump())
        m=args.model;t=args.train
        normalized=dict(layers=m.num_layers,hidden=m.hidden_size,ffn=m.ffn_hidden_size,q=m.num_attention_heads,
            kv=m.num_query_groups,head_dim=m.kv_channels,vocab=m.padded_vocab_size,eps=m.norm_epsilon,
            rope_base=m.rotary_base,rope_scaling=False,rope_interpolation=m.rotary_seq_len_interpolation_factor,
            qkv_bias=m.add_qkv_bias,linear_bias=m.add_bias_linear,untied=m.untie_embeddings_and_output_weights,
            qk_rmsnorm=m.qk_layernorm,normalization=m.normalization,swiglu=m.gated_linear_unit,
            seq=t.seq_length,gbs=t.global_batch_size,dtype=args.parallel.mixed_precision,
            hidden_dropout=m.hidden_dropout,attention_dropout=m.attention_dropout,lr=t.lr,min_lr=t.min_lr,
            eps_adam=t.adam_eps,betas=[t.adam_beta1,t.adam_beta2],weight_decay=t.weight_decay,
            clip_grad=t.clip_grad,init_std=t.init_method_std,seed=t.seed,reduce_fp32=args.parallel.reduce_in_fp32)
    else:
        from megatron.training.arguments import parse_args,validate_args,core_transformer_config_from_args
        sys.argv=['h20-audit']+meg_arguments(c,folder)
        os.environ.update(RANK='0',WORLD_SIZE='8')
        args=validate_args(parse_args())
        config=core_transformer_config_from_args(args)
        # Explicit same initializer for residual output weights as native Galvatron.
        config.output_layer_init_method=config.init_method
        save(folder/'arguments.json',sys.argv[1:]);save(folder/'effective.json',vars(args))
        save(folder/'transformer_config.json',vars(config))
        normalized=dict(layers=args.num_layers,hidden=args.hidden_size,ffn=args.ffn_hidden_size,
            q=args.num_attention_heads,kv=args.num_query_groups,head_dim=args.kv_channels,
            vocab=args.padded_vocab_size,eps=args.norm_epsilon,rope_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,rope_interpolation=args.rotary_seq_len_interpolation_factor,
            qkv_bias=args.add_qkv_bias,linear_bias=args.add_bias_linear,untied=args.untie_embeddings_and_output_weights,
            qk_rmsnorm=args.qk_layernorm,normalization=args.normalization,swiglu=args.swiglu,
            seq=args.seq_length,gbs=args.global_batch_size,dtype='fp16' if args.fp16 else 'bf16',
            hidden_dropout=args.hidden_dropout,attention_dropout=args.attention_dropout,lr=args.lr,min_lr=args.min_lr,
            eps_adam=args.adam_eps,betas=[args.adam_beta1,args.adam_beta2],weight_decay=args.weight_decay,
            clip_grad=args.clip_grad,init_std=args.init_method_std,seed=args.seed,reduce_fp32=args.accumulate_allreduce_grads_in_fp32)
    normalized.update(weight_decay_scope='all_parameters',output_layer_init_std=normalized['init_std'],
                      loss='causal next-token cross-entropy, float32 logits',optimizer='AdamW with float32 master parameters/states')
    save(folder/'normalized.json',normalized)
    print('EXPORTED',a.system,name,flush=True)
