# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
from collections import defaultdict
import copy
import random
from dataclasses import dataclass, field, make_dataclass
from enum import Enum, auto
from functools import reduce
import math
import hashlib
import operator
import os
import re
import sys
import subprocess
import time
from typing import Optional, Tuple, Dict, List, Union
import numpy as np
import torch
import json
import atexit
import torch.distributed
import threading
import pandas as pd
import glob
import warnings
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import argparse
import itertools
import fcntl

SIZE_BF16=2
SIZE_FP32=4
ITERATION_LOOP_TIME = int(os.getenv("DTSIR_PROFILE_ITERS", "50"))
WARMUP_LOOP_TIME = int(os.getenv("DTSIR_PROFILE_WARMUP", "50"))
BAND_WIDTH_UNIDIRECTIONAL = 25*1024*1024*1024 # 字节/s，这是查的带宽，实际过程中小多了
#BAND_WIDTH_MEMORY_TRANS=31.25*1024*1024*1024#字节/s
BAND_WIDTH_MEMORY_TRANS=20*1024*1024*1024#字节/s
NPU_TFLOPS=320*1e12#FLOPS(910B)
GPU_TFLOPS=320*1e12#FLOPS(A100)


# ========================= TPDS experiment framework =========================
# The framework is intentionally lightweight: it keeps the original model/search
# implementation and adds dependency-aware reuse, instrumentation, and experiment
# routing controlled entirely through environment variables.
from collections import Counter


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_list_int(name: str, default=None):
    value = os.getenv(name)
    if not value:
        return list(default or [])
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)



def _tpds_strategy_distance(a: dict, b: dict) -> int:
    """Hamming-style distance over strategy choices for small, diverse smoke sets."""
    keys = sorted(set(a) | set(b))
    return sum(1 for k in keys if _jsonable(a.get(k)) != _jsonable(b.get(k)))


def _tpds_select_strategy_combos(combos, limit, seed=2026, mode="diverse"):
    """Select a deterministic subset of strategy combinations.

    `diverse` is intended for smoke/ablation runs: it starts from the most
    disabled/baseline-like combination and greedily maximizes strategy Hamming
    distance, so a small limit still exercises multiple strategy deltas.
    """
    combos = list(combos)
    if limit <= 0 or len(combos) <= limit:
        return combos
    mode = (mode or "diverse").lower()
    rng = random.Random(seed)
    if mode == "first":
        return combos[:limit]
    if mode == "random":
        idx = sorted(rng.sample(range(len(combos)), limit))
        return [combos[i] for i in idx]

    hybrid_values = [c.get("Hybrid_MHA_MQA") for c in combos]
    numeric_hybrid = [v for v in hybrid_values if isinstance(v, (int, float))]
    # In set_opticonfig, num_query_groups == num_attention_heads means ordinary MHA/off.
    # The largest enumerated numeric value is therefore the baseline-like choice.
    hybrid_off_value = max(numeric_hybrid) if numeric_hybrid else None

    def baseline_score(c):
        score = 0
        if c.get("ReCompute") is None:
            score += 1
        if c.get("VirtualPipe") is None:
            score += 1
        if not bool(c.get("DistributedOptimizer", False)):
            score += 1
        hv = c.get("Hybrid_MHA_MQA")
        if hv is None or (hybrid_off_value is not None and hv == hybrid_off_value):
            score += 1
        return score

    first_idx = max(range(len(combos)), key=lambda i: (baseline_score(combos[i]), -i))
    selected = [combos[first_idx]]
    remaining = [c for i, c in enumerate(combos) if i != first_idx]
    while remaining and len(selected) < limit:
        scored = []
        for i, c in enumerate(remaining):
            mind = min(_tpds_strategy_distance(c, x) for x in selected)
            # Secondary score favors combinations that turn on/off different mechanisms.
            scored.append((mind, baseline_score(c), rng.random(), i))
        _, _, _, pick_i = max(scored)
        selected.append(remaining.pop(pick_i))
    return selected


@dataclass
class TPDSExperimentConfig:
    experiment: str = ""
    variant: str = "full"
    run_id: str = ""
    log_dir: str = ""
    candidate_limit: int = 0
    strategy_limit: int = 0
    strategy_selection: str = "diverse"
    max_mbs: int = 0
    sample_seed: int = 2026
    trace_every: int = 1
    capture_candidates: bool = False
    capture_rejected: bool = False
    profile_seed_from_existing: bool = False
    single_gpu_smoke: bool = False

    def refresh(self, mmlogs_path: str = ""):
        self.experiment = os.getenv("DTSIR_EXPERIMENT", "").strip().lower()
        self.variant = os.getenv("DTSIR_VARIANT", "full").strip().lower()
        self.run_id = os.getenv("DTSIR_RUN_ID", "") or datetime.now().strftime("%Y%m%d_%H%M%S")
        default_dir = os.path.join(mmlogs_path or ".", "tpds_experiments")
        self.log_dir = os.getenv("DTSIR_LOG_DIR", default_dir)
        self.candidate_limit = _env_int("DTSIR_MAX_CANDIDATES", 0)
        self.strategy_limit = _env_int("DTSIR_MAX_STRATEGY_COMBOS", 0)
        self.strategy_selection = os.getenv("DTSIR_STRATEGY_SELECTION", "diverse").strip().lower()
        self.max_mbs = _env_int("DTSIR_MAX_MBS", 0)
        self.sample_seed = _env_int("DTSIR_SAMPLE_SEED", 2026)
        self.trace_every = max(1, _env_int("DTSIR_TRACE_EVERY", 1))
        self.capture_candidates = _env_bool("DTSIR_CAPTURE_CANDIDATES", self.experiment in {"ranking", "oracle", "compound"})
        self.capture_rejected = _env_bool("DTSIR_CAPTURE_REJECTED", self.experiment == "compound")
        self.profile_seed_from_existing = _env_bool("DTSIR_PROFILE_SEED_FROM_EXISTING", False)
        self.single_gpu_smoke = _env_bool("DTSIR_SINGLE_GPU_SMOKE", False)
        os.makedirs(self.log_dir, exist_ok=True)
        return self

    @property
    def active(self) -> bool:
        return self.experiment not in {"", "off", "none", "measure"}

    @property
    def use_incremental(self) -> bool:
        override = os.getenv("DTSIR_INCREMENTAL")
        if override is not None:
            return _env_bool("DTSIR_INCREMENTAL")
        return self.variant in {"full", "no_profile"}

    @property
    def use_profile_reuse(self) -> bool:
        override = os.getenv("DTSIR_PROFILE_REUSE")
        if override is not None:
            return _env_bool("DTSIR_PROFILE_REUSE")
        return self.variant in {"full", "no_inc"}


class TPDSExperimentRuntime:
    STRATEGY_TO_ANALYSES = {
        "ReCompute": {"memory", "schedule"},
        "VirtualPipe": {"memory", "schedule"},
        # In the current implementation, distributed optimizer affects persistent
        # memory ownership; DP_flowgraph intentionally contributes no time penalty.
        "DistributedOptimizer": {"memory"},
        "Hybrid_MHA_MQA": {"shape", "layout", "memory", "profile", "schedule", "structural_space"},
    }

    def __init__(self):
        self.config = TPDSExperimentConfig()
        self.stats = Counter()
        self.timings = Counter()
        self.structural_space_cache = {}
        self.memory_template_cache = {}
        self.cost_cache = {}
        self.event_cost_cache = {}
        self.profile_overlay = {}
        self.profile_unique_measured = set()
        self.comm_query_conditions = set()
        self.candidate_records = []
        self.rejected_records = []
        self.trace = []
        self.current_strategy = {}
        self.previous_strategy = None
        self._candidate_counter = 0

    def refresh(self, mmlogs_path: str = ""):
        self.config.refresh(mmlogs_path)
        return self

    def reset(self, clear_incremental=True, clear_profile=True, clear_records=True):
        self.stats = Counter()
        self.timings = Counter()
        self.trace = []
        self.comm_query_conditions = set()
        self.previous_strategy = None
        self.current_strategy = {}
        self._candidate_counter = 0
        if clear_incremental:
            self.structural_space_cache = {}
            self.memory_template_cache = {}
            self.cost_cache = {}
            self.event_cost_cache = {}
        if clear_profile:
            self.profile_overlay = {}
            self.profile_unique_measured = set()
        if clear_records:
            self.candidate_records = []
            self.rejected_records = []

    def set_variant(self, variant: str):
        self.config.variant = variant

    def strategy_snapshot(self, args):
        recompute_modules = getattr(args, "recompute_modules", None)
        if isinstance(recompute_modules, list):
            recompute_modules = tuple(recompute_modules)
        return {
            "ReCompute": (getattr(args, "recompute_granularity", None), recompute_modules),
            "VirtualPipe": getattr(args, "num_layers_per_virtual_pipeline_stage", None),
            "DistributedOptimizer": bool(getattr(args, "use_distributed_optimizer", False)),
            "Hybrid_MHA_MQA": (
                bool(getattr(args, "group_query_attention", False)),
                getattr(args, "num_query_groups", None),
            ),
        }

    def note_strategy(self, args):
        snap = self.strategy_snapshot(args)
        self.current_strategy = snap
        self.stats["strategy_candidates"] += 1
        if self.previous_strategy is None:
            changed = list(snap.keys())
        else:
            changed = [k for k in snap if snap[k] != self.previous_strategy.get(k)]
        affected = set()
        for key in changed:
            affected.update(self.STRATEGY_TO_ANALYSES.get(key, set()))
        self.stats["strategy_changed_facts"] += len(changed)
        self.stats["strategy_affected_analysis_types"] += len(affected)
        self.previous_strategy = copy.deepcopy(snap)
        return changed, affected

    def analysis(self, name: str, seconds: float = 0.0, cache_hit: bool = False, count: int = 1):
        self.stats[f"analysis_{name}_requests"] += count
        if cache_hit:
            self.stats[f"analysis_{name}_cache_hit"] += count
        else:
            self.stats[f"analysis_{name}"] += count
            self.stats["analysis_total"] += count
        if seconds:
            self.timings[f"analysis_{name}_seconds"] += seconds

    def profile_value(self, model, calc_name, calc_key, measure_fn):
        self.stats["profile_queries"] += 1
        full_key = (calc_name, calc_key)
        if self.config.use_profile_reuse:
            if full_key in self.profile_overlay:
                self.stats["profile_hits"] += 1
                return self.profile_overlay[full_key]
            if self.config.profile_seed_from_existing and hasattr(model, "map_manager"):
                bucket = model.map_manager.data_map.get(calc_name, {})
                value = bucket.get(calc_key, 0)
                if value not in (0, None):
                    self.profile_overlay[full_key] = value
                    self.stats["profile_hits"] += 1
                    self.stats["profile_seed_hits"] += 1
                    return value
        self.stats["profile_misses"] += 1
        t0 = time.perf_counter()
        value = measure_fn()
        self.timings["profile_measure_seconds"] += time.perf_counter() - t0
        self.profile_unique_measured.add(full_key)
        self.stats["profile_measurements"] += 1
        if self.config.use_profile_reuse:
            self.profile_overlay[full_key] = value
        return value

    def record_candidate(self, record: dict):
        self._candidate_counter += 1
        self.stats["candidates_evaluated"] += 1
        if self.config.capture_candidates:
            self.candidate_records.append(_jsonable(record))
        if self._candidate_counter % self.config.trace_every == 0:
            q = self.stats["profile_queries"]
            hits = self.stats["profile_hits"]
            self.trace.append({
                "candidate": self._candidate_counter,
                "profile_queries": q,
                "profile_hits": hits,
                "profile_hit_rate": (hits / q) if q else 0.0,
                "profile_measurements": self.stats["profile_measurements"],
                "unique_profile_measurements": len(self.profile_unique_measured),
                "analysis_total": self.stats["analysis_total"],
            })

    def record_rejected(self, record: dict):
        self.stats["candidates_rejected_memory"] += 1
        if self.config.capture_rejected:
            self.rejected_records.append(_jsonable(record))

    def result_dict(self, extra=None):
        q = self.stats["profile_queries"]
        hits = self.stats["profile_hits"]
        out = {
            "experiment": self.config.experiment,
            "variant": self.config.variant,
            "run_id": self.config.run_id,
            "incremental": self.config.use_incremental,
            "profile_reuse": self.config.use_profile_reuse,
            "stats": dict(self.stats),
            "timings": dict(self.timings),
            "profile_hit_rate": (hits / q) if q else 0.0,
            "unique_profile_measurements": len(self.profile_unique_measured),
            "trace": self.trace,
            "unique_communication_queries": len(self.comm_query_conditions),
            "communication_query_conditions": sorted(self.comm_query_conditions),
            "timing_note": "Analysis timers are inclusive and may contain Profile measurements; do not sum them.",
        }
        if extra:
            out.update(_jsonable(extra))
        return out

    def write_result(self, name: str, payload: dict):
        os.makedirs(self.config.log_dir, exist_ok=True)
        path = os.path.join(self.config.log_dir, f"{self.config.run_id}_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)
        return path


TPDS_RUNTIME = TPDSExperimentRuntime()
# ============================================================================

class FileJSONHandler:
    def __init__(self, file_path,init_flag=True,print_flag=False):
        self.file_path = file_path
        self.lock_fd = None  # 文件级锁
        self._initialize_file()
        self.data_map={}
        self.print_flag=print_flag
        if init_flag:
            self.data_map = self._load_from_json()
        else:
            if os.path.exists(self.file_path) and os.path.isfile(self.file_path):os.remove(self.file_path)
        self.sonmap=[]
        self.value=0
        # 注册退出时的保存操作
        atexit.register(self._save_to_json)
        if print_flag:
            print(f"已从 {file_path} 加载数据，包含 {len(self.data_map)} 个条目")
    def _initialize_file(self):
        """初始化文件，确保文件存在"""
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                f.write('{}')  # 创建空JSON对象

    def _acquire_lock(self, timeout=5):
        """获取文件锁（带超时）"""
        start_time = time.time()
        self.lock_fd = open(self.file_path, 'a')  # 打开文件获取文件描述符
        while True:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                # if timeout and (time.time() - start_time) > timeout:
                #     self.lock_fd.close()
                #     return False
                time.sleep(0.01)

    def _release_lock(self):
        """释放文件锁"""
        if self.lock_fd:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
            self.lock_fd.close()

    def _load_from_json(self):
        """安全读取JSON数据"""
        try:
            if not self._acquire_lock():
                print("获取锁超时")
                return {}
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            with open(self.file_path, 'w', encoding='utf-8') as file:
                json.dump({}, file, indent=4, ensure_ascii=False)
                print(f"文件不存在，已创建新文件：{self.file_path}")
                return {}
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}，创建新字典")
            return {}
        except Exception as e:
            print(f"加载错误: {e}，创建新字典")
            return {}
        finally:
            self._release_lock()

    def _save_to_json(self, indent=4):
        """安全写入JSON数据"""
        if not self._acquire_lock():
            print("获取锁超时")
            return False
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.data_map, f, indent=indent, ensure_ascii=False)
            if self.print_flag:
                print(f"数据已保存至 {self.file_path}")
            return True
        except Exception as e:
            print(f"写入错误: {e}")
            return False
        finally:
            self._release_lock()

class CommDataProcessor:
    
    def __init__(self, folder_path, print_flag=False, force_reprocess=False):
        self.print_flag = print_flag
        self.folder_path=folder_path
        self.result = {}

        # 检查是否已存在profile_comm.json且不需要强制重新处理
        profile_comm_path = os.path.join(folder_path, 'profile_comm.json')
        
        if not force_reprocess and os.path.exists(profile_comm_path):
            if self.print_flag:
                print(f"检测到已存在的profile_comm.json，直接加载...")
            self.load_comm_model(profile_comm_path)
            if self.print_flag:
                print(f"成功加载现有模型，包含 {len(self.result)} 个操作类型")
        else:
            if self.print_flag and force_reprocess:
                print("强制重新处理数据...")
            elif self.print_flag:
                print("未找到profile_comm.json，开始处理数据...")
            self.result = self.process_folder(folder_path)

    def michaelis_menten(self, x, vmax, km):
        """Michaelis-Menten饱和模型公式"""
        return vmax * x / (km + x)

    def detect_format(self, lines):
        """检测文件格式"""
        for line in lines:
            if 'data_size(Bytes)' in line and 'aveg_time' in line and 'alg_bandwidth' in line:
                return 'format1'  # 原始格式
            elif ('size' in line and 'count' in line and 'time' in line and 
                  'algbw' in line and 'busbw' in line):
                return 'format2'  # 新格式
        # 如果无法检测，尝试基于内容推断
        for line in lines:
            if '|' in line and 'data_size' in line:
                return 'format1'
            elif not line.startswith('#') and len(line.split()) >= 6:
                return 'format2'
        return 'format1'  # 默认格式

    def extract_data_format1(self, lines):
        """提取格式1的数据"""
        data_lines = [line.strip() for line in lines if '|' in line and 'data_size' not in line]
        
        data = []
        for line in data_lines:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            try:
                if len(parts) >= 3:  # 放宽条件，不一定需要success列
                    data.append({
                        'data_size': int(parts[0]),
                        'time': float(parts[1]),
                        'bandwidth': float(parts[2])
                    })
            except Exception as e:
                if self.print_flag:
                    print(f"格式1解析错误: {line} - {str(e)}")
        
        return data

    def extract_data_format2(self, lines):
        """提取格式2的数据"""
        data = []
        header_found = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # 查找包含列名的header行
            if not header_found and 'size' in line and 'time' in line and 'algbw' in line:
                header_found = True
                if self.print_flag:
                    print(f"找到header行: {line}")
                continue
            
            # 跳过注释行和空行
            if line.startswith('#') or not line:
                continue
                
            # 处理数据行
            parts = line.split()
            
            try:
                # 验证数据格式：至少包含6列
                if len(parts) >= 6:
                    data_size = int(parts[0])
                    
                    # 使用固定位置（基于常见格式）
                    if len(parts) >= 9:
                        time_idx = 5  # 第6列
                        algbw_idx = 6  # 第7列
                    elif len(parts) >= 7:
                        time_idx = 5  # 第6列
                        algbw_idx = 6  # 第7列
                    else:
                        continue
                    
                    time = float(parts[time_idx])
                    algbw = float(parts[algbw_idx])
                    
                    # 过滤掉异常小的数据（如size=0）
                    if data_size > 0 and time > 0 and algbw >= 0:
                        data.append({
                            'data_size': data_size,
                            'time': time,
                            'bandwidth': algbw
                        })
                        
                        if self.print_flag and len(data) <= 3:
                            print(f"成功解析: size={data_size}, time={time}, algbw={algbw}")
                    
            except (ValueError, IndexError) as e:
                if self.print_flag and i < 10:
                    print(f"跳过行 {i}: {line} - 错误: {str(e)}")
                continue
        
        if self.print_flag:
            print(f"格式2解析完成，找到 {len(data)} 条数据")
        
        return data

    def extract_data(self, file_path):
        """从txt文件中提取数据并验证"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                if self.print_flag:
                    print(f"文件为空: {file_path}")
                return pd.DataFrame()
            
            # 检测文件格式
            file_format = self.detect_format(lines)
            if self.print_flag:
                print(f"检测到文件格式: {file_format}")
            
            # 根据格式选择提取方法
            if file_format == 'format1':
                data_list = self.extract_data_format1(lines)
            else:
                data_list = self.extract_data_format2(lines)
            
            # 转换为DataFrame
            if data_list:
                df = pd.DataFrame(data_list)
                if self.print_flag:
                    print(f"成功提取 {len(df)} 条数据")
                return df
            else:
                if self.print_flag:
                    print(f"未提取到数据: {file_path}")
                return pd.DataFrame()
                
        except Exception as e:
            if self.print_flag:
                print(f"读取文件错误 {file_path}: {str(e)}")
            return pd.DataFrame()

    def fit_nonlinear_model(self, data):
        """使用Michaelis-Menten非线性回归模型"""
        if len(data) < 3:
            if self.print_flag:
                print("数据点不足，无法进行非线性回归")
            # 使用简单线性插值作为备选方案
            return self.fit_simple_model(data)
        
        X = data['data_size'].values
        y = data['bandwidth'].values
        
        # 获取初始参数估计
        vmax_guess = max(y) * 1.1
        km_guess = np.median(X)
        
        # 拟合非线性模型
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                params, _ = curve_fit(self.michaelis_menten, X, y, 
                                    p0=[vmax_guess, km_guess], maxfev=5000)
                vmax = params[0]
                km = params[1]
                
                # 计算R²评分
                y_pred = self.michaelis_menten(X, vmax, km)
                r2 = r2_score(y, y_pred)
                
            except (RuntimeError, ValueError) as e:
                if self.print_flag:
                    print(f"非线性拟合失败，使用备选方案: {str(e)}")
                return self.fit_simple_model(data)
        
        # 创建预测函数（带真实值上限）
        max_real_bandwidth = max(y)
        
        def predict_with_cap(x):
            pred = self.michaelis_menten(x, vmax, km)
            return np.minimum(pred, max_real_bandwidth)
        
        return {
            'model_type': 'Michaelis-Menten',
            'vmax': float(vmax),
            'km': float(km),
            'max_bandwidth': float(max_real_bandwidth),
            'r2_score': float(r2),
            'predict_fn': predict_with_cap
        }

    def fit_simple_model(self, data):
        """备选方案：简单模型"""
        X = data['data_size'].values
        y = data['bandwidth'].values
        
        max_bw = max(y)
        median_size = np.median(X)
        
        def simple_predict(x):
            # 简单线性插值
            if len(data) == 1:
                return np.full_like(x, y[0])
            else:
                # 使用最大带宽的简单饱和模型
                return np.minimum(max_bw * x / (median_size + x), max_bw)
        
        # 计算简单模型的R²
        y_pred = simple_predict(X)
        r2 = r2_score(y, y_pred) if len(y) > 1 else 1.0
        
        return {
            'model_type': 'Simple-Saturation',
            'vmax': float(max_bw),
            'km': float(median_size),
            'max_bandwidth': float(max_bw),
            'r2_score': float(r2),
            'predict_fn': simple_predict
        }

    def validate_model(self, data, model):
        """验证模型并对比真实值"""
        predictions = []
        for _, row in data.iterrows():
            data_size = row['data_size']
            true_bandwidth = row['bandwidth']
            true_time = row['time']
            
            # 预测带宽和时间
            pred_bandwidth = model['predict_fn'](data_size)
            pred_time = data_size / (pred_bandwidth * 1000) if pred_bandwidth > 0 else float('inf')
            
            # 计算误差
            bw_error = abs(pred_bandwidth - true_bandwidth)
            time_error = abs(pred_time - true_time)
            
            predictions.append({
                'data_size': int(data_size),
                'true_bandwidth': true_bandwidth,
                'pred_bandwidth': pred_bandwidth,
                'bandwidth_error': bw_error,
                'true_time': true_time,
                'pred_time': pred_time,
                'time_error': time_error,
                'time_error_pct': (time_error / true_time) * 100 if true_time > 0 else float('inf')
            })
        
        # 计算平均误差
        valid_predictions = [p for p in predictions if p['time_error_pct'] != float('inf')]
        if valid_predictions:
            avg_bw_error = np.mean([p['bandwidth_error'] for p in valid_predictions])
            avg_time_error = np.mean([p['time_error'] for p in valid_predictions])
            avg_time_error_pct = np.mean([p['time_error_pct'] for p in valid_predictions])
        else:
            avg_bw_error = avg_time_error = avg_time_error_pct = 0
        
        return predictions, avg_bw_error, avg_time_error, avg_time_error_pct

    def process_folder(self, folder_path, jsonname='profile_comm'):
        """处理文件夹中的所有txt文件"""
        results = {}
        validation_results = {}
        
        # 获取所有txt文件
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        
        if not txt_files:
            if self.print_flag:
                print(f"在文件夹 {folder_path} 中未找到txt文件")
            return results
        
        if self.print_flag:
            print(f"找到 {len(txt_files)} 个txt文件")
        
        for file_path in txt_files:
            filename = os.path.basename(file_path)
            file_key = os.path.splitext(filename)[0]
            
            if self.print_flag:
                print(f"\n处理文件: {filename}")
            
            # 提取数据
            data = self.extract_data(file_path)
            if data.empty:
                if self.print_flag:
                    print(f"跳过空文件: {filename}")
                continue
            
            if self.print_flag:
                print(f"提取到 {len(data)} 行数据")
            
            # 训练模型
            model_info = self.fit_nonlinear_model(data)
            
            # 验证模型
            predictions, avg_bw_error, avg_time_error, avg_time_error_pct = self.validate_model(data, model_info)
            
            # 生成操作类型键
            op_key = file_key.replace('_test', '').replace('_perf', '').replace('all_gather', 'all_gather')
            
            # 保存到结果集
            results[op_key] = {
                'model_type': model_info['model_type'],
                'vmax': model_info['vmax'],
                'km': model_info['km'],
                'max_bandwidth': model_info['max_bandwidth'],
                'r2_score': model_info['r2_score']
            }
            
            # 保存验证结果
            validation_results[op_key] = {
                'predictions': predictions,
                'avg_bandwidth_error': avg_bw_error,
                'avg_time_error': avg_time_error,
                'avg_time_error_percentage': avg_time_error_pct
            }
            
            if self.print_flag:
                print(f"成功为 {op_key} 创建模型: Vmax={model_info['vmax']:.2f} GB/s, R²={model_info['r2_score']:.4f}")
        
        # 保存JSON结果
        if results:
            file_path = f'{folder_path}/{jsonname}.json'
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # 保存验证结果到单独文件
            with open(f'{folder_path}/validation_results.json', 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            if self.print_flag:
                print(f"\n处理完成! 成功处理 {len(results)} 个文件")
                print(f"结果已保存到:")
                print(f"- {folder_path}/{jsonname}.json: 模型参数")
                print(f"- {folder_path}/validation_results.json: 验证结果")
        else:
            if self.print_flag:
                print("警告: 未成功处理任何文件，JSON文件为空")
        
        return results

    def load_comm_model(self, json_path):
        """加载通信模型参数"""
        try:
            with open(json_path, 'r') as f:
                self.result = json.load(f)
        except Exception as e:
            print(f'profile_comm.json load error! error reason is :{e}')

    def predict_bandwidth(self, model_params, data_size):
        """使用Enhanced Michaelis-Menten模型预测带宽"""
        vmax = model_params['vmax']
        km = model_params['km']
        bandwidth = vmax * data_size / (km + data_size)
        return min(bandwidth, model_params['max_bandwidth'])

    def predict_time(self, data_size, bandwidth):
        """根据带宽预测时间(μs)"""
        if bandwidth <= 0:
            return float('inf')
        return data_size / (bandwidth * 1000)  # 时间(μs)=数据大小/(带宽*1000)

    def print_process_result(self):  
        if self.result:
            first_file = next(iter(self.result.keys()))
            model_info = self.result[first_file]
            print("\n模型参数示例:")
            print(f"- 模型类型: {model_info['model_type']}")
            print(f"- Vmax: {model_info['vmax']:.2f} GB/s (最大带宽)")
            print(f"- Km: {model_info['km']:.0f} bytes (半饱和值)")
            print(f"- 实际最大带宽: {model_info['max_bandwidth']:.2f} GB/s")
            print(f"- R²得分: {model_info['r2_score']:.4f}")

    def data_predict(self, opsize: list, read_flag=False, json_path=''):
        # 加载模型
        op_type = opsize[0]
        data_size = int(opsize[1])
        rt = globals().get('TPDS_RUNTIME')
        if rt is not None and rt.config.active:
            rt.stats['communication_queries'] += 1
            rt.comm_query_conditions.add((str(op_type), data_size))
        if read_flag:
            self.load_comm_model(json_path)
        model = self.result
        if op_type == 'recv':
            #op_type = 'reduce' npu下
            op_type = 'sendrecv'
        if op_type == 'send':
            #op_type = 'scatter' npu下
            op_type = 'sendrecv'
        if op_type == '' or data_size == 0: 
            return 0
        if op_type not in model:
            if rt is not None and rt.config.active:
                rt.stats['invalid_communication_queries'] += 1
            valid_ops = ", ".join(model.keys())
            print(f"错误：无效操作类型 '{op_type}'。有效类型: {valid_ops}")
            return 0
        
        op_params = model[op_type]
        bandwidth = self.predict_bandwidth(op_params, data_size)
        time = self.predict_time(data_size, bandwidth)
        return time

PP_Form=[{"name":"1F1B","acti_form":"pp-i"}]

def split_with_offset(a, n_splits, start_offset=1):
    length = len(a)
    base_size = length // n_splits
    extra = length % n_splits
    length_list = [base_size for _ in range(n_splits)]

    for _ in range(extra):
        if start_offset >= len(length_list): start_offset -= len(length_list)
        length_list[start_offset] += 1
        start_offset += 1

    result = []
    start_idx = 0
    for d in length_list:
        result.append(a[start_idx:start_idx+d])
        start_idx += d
    return result

#诸元素相乘

def Calc_Sub(left:Tuple,right:Tuple):
    #print(f'mask{left+mask}')
    if torch.cuda.is_available():
        x=torch.rand(left,dtype=torch.float16,device='cuda')
        y=torch.rand(right, dtype=torch.float16,device='cuda')
    else:
        x=torch.rand(left,dtype=torch.float16, device='cpu')
        y=torch.rand(right, dtype=torch.float16,device='cpu')
    for _ in range(WARMUP_LOOP_TIME):
        torch.sub(x,y)
    torch.cuda.synchronize()  

    start_time = time.time()
    for i in range(ITERATION_LOOP_TIME):
        torch.sub(x,y)
    torch.cuda.synchronize()   
    result=(time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME
    return result

def Calc_Mask(left:Tuple,mask:Tuple):
    if torch.cuda.is_available():
        x=torch.rand(left,dtype=torch.float16,device='cuda')
    else:
        x=torch.rand(left,dtype=torch.float16, device='cpu')
    causal_mask=torch.triu(x).bool()
    for _ in range(WARMUP_LOOP_TIME):
        x.masked_fill(causal_mask,float('-inf'))
    torch.cuda.synchronize()  

    start_time = time.time()
    for i in range(ITERATION_LOOP_TIME):
        x.masked_fill(causal_mask,float('-inf'))
    torch.cuda.synchronize()   
    result=(time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME
    return result

def Calc_Mul(left:Tuple,right:Tuple):
    #print(f'mul{left+right}')
    if torch.cuda.is_available():
        x=torch.rand(left, dtype=torch.float16,device='cuda')
        y=torch.rand(right, dtype=torch.float16,device='cuda')
    else:
        x=torch.rand(left, dtype=torch.float16,device='cpu')
        y=torch.rand(right, dtype=torch.float16,device='cpu')
    
    for _ in range(WARMUP_LOOP_TIME):
        torch.mul(x, y)
    torch.cuda.synchronize()

    start_time = time.time()
    for i in range(ITERATION_LOOP_TIME):
        torch.mul(x, y)
    torch.cuda.synchronize()   
    result=(time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME
    return result

def Calc_Dropout(left:Tuple,radio:float):
    #print(f'dropout{left}')
    if torch.cuda.is_available():
        x=torch.rand(left,dtype=torch.float16,device='cuda')
    else:
        x=torch.rand(left,dtype=torch.float16, device='cpu')
    dropout=torch.nn.Dropout(p=radio)
    for _ in range(WARMUP_LOOP_TIME):
        dropout(x)
    torch.cuda.synchronize()  

    start_time = time.time()
    for i in range(ITERATION_LOOP_TIME):
        dropout(x)
    torch.cuda.synchronize()   
    result=(time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME
    return result

def Calc_SDPA(q_list:Tuple,k_list:Tuple,v_list:Tuple):
    #gpu版本的flash_attn需要手动对齐k，v的head
    q = torch.randn(q_list[0], q_list[1], q_list[2], q_list[3], device='cuda', dtype=torch.float16)
    k = torch.randn(k_list[0], q_list[1], k_list[3], k_list[2], device='cuda', dtype=torch.float16)#算子内部自己转置
    v = torch.randn(v_list[0], q_list[1], v_list[2], v_list[3], device='cuda', dtype=torch.float16)
    # q = torch.randn(q_list[0], q_list[1], q_list[2], q_list[3], device='cuda', dtype=torch.float16)
    # k = torch.randn(k_list[0], k_list[1], k_list[3], k_list[2], device='cuda', dtype=torch.float16)#算子内部自己转置
    # v = torch.randn(v_list[0], v_list[1], v_list[2], v_list[3], device='cuda', dtype=torch.float16)
        # try:
        #output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
            #print("成功使用 FlashAttention 后端。")
        # except RuntimeError as e:
        #     print(f"FlashAttention 不可用: {e}")

    for _ in range(WARMUP_LOOP_TIME):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v,attn_mask=None, dropout_p=0.0,is_causal=True)
        #attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v,attn_mask=None, dropout_p=0.0,is_causal=True)#可选的注意力掩码和指定因果掩码
    torch.cuda.synchronize()  

    start_time = time.time()
    for i in range(ITERATION_LOOP_TIME):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v,attn_mask=None, dropout_p=0.0,is_causal=True)
        #attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v,attn_mask=None, dropout_p=0.0,is_causal=True)#可选的注意力掩码和指定因果掩码
    torch.cuda.synchronize()   
    result=(time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME
    return result

def Calc_Matmul(left:Tuple,right:Tuple):
    #print(f'matmul{left+right}')
    if torch.cuda.is_available():
        x=torch.rand(left,dtype=torch.float16,device='cuda')
        y=torch.rand(right,dtype=torch.float16, device='cuda')
    else:
        x=torch.rand(left,dtype=torch.float16, device='cpu')
        y=torch.rand(right,dtype=torch.float16, device='cpu')
    for _ in range(WARMUP_LOOP_TIME):
        torch.matmul(x, y)
    torch.cuda.synchronize()  

    start_time = time.time()
    for i in range(ITERATION_LOOP_TIME):
        torch.matmul(x, y)
    torch.cuda.synchronize()   
    result=(time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME
    return result

def Calc_LayerNorm(left:Tuple,hidden_size):
    #print(f'LayerNorm{left+hidden_size}')
    class LayerNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            super(LayerNorm, self).__init__()
            self.layer_norm = torch.nn.LayerNorm(normalized_shape=hidden_size,dtype=torch.float16,eps=eps)
        def forward(self, x):
            if self.layer_norm.weight.dtype != x.dtype:
                x = x.to(self.layer_norm.weight.dtype)  # 将x转换为gamma的数据类型
            return self.layer_norm(*x)
    abc=LayerNorm(hidden_size)
    if torch.cuda.is_available():
        x=torch.rand(left,dtype=torch.float16, device='cuda')
    else:
        x=torch.rand(left,dtype=torch.float16, device='cpu')
    for _ in range(WARMUP_LOOP_TIME):
        abc.forward(x)
    torch.cuda.synchronize()

    start_time = time.time()
    for i in range(ITERATION_LOOP_TIME):
        abc.forward(x)
    torch.cuda.synchronize()   
    result=(time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME
    return result

def Calc_RmsNorm(left:Tuple,hidden_size):
    #print(f'RmsNorm{left+hidden_size}')
    class FusedRmsNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-6) -> None:
            super().__init__()
            #self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float16)).npu()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))
            self.eps = eps
        def forward(self, x):
            if self.weight.device != x.device:
                self.weight.data = self.weight.data.to(x.device)
            if self.weight.dtype != x.dtype:
                x = x.to(self.weight.dtype)  # 将x转换为gamma的数据类型
            rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return self.weight * x * rms
            #return torch_npu.npu_rms_norm(x[0], self.weight, epsilon=self.eps)[0]
    abc=FusedRmsNorm(hidden_size)
    if torch.cuda.is_available():
        x=torch.rand(left,dtype=torch.float16, device='cuda')
    else:
        x=torch.rand(left,dtype=torch.float16, device='cpu')
    for _ in range(WARMUP_LOOP_TIME):
        abc.forward(x)
    torch.cuda.synchronize()

    start_time = time.time()
    for i in range(ITERATION_LOOP_TIME):
        abc.forward(x)
    torch.cuda.synchronize()   
    result=(time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME
    return result

def Calc_Softmax(left:Tuple):
    #print(f'Softmax{left}')
    if torch.cuda.is_available():
        x=torch.rand(left,dtype=torch.float16, device='cuda')
    else:
        x=torch.rand(left,dtype=torch.float16, device='cpu')
    for _ in range(WARMUP_LOOP_TIME):
        torch.nn.functional.softmax(x)
    torch.cuda.synchronize()

    start_time = time.time()
    for i in range(ITERATION_LOOP_TIME):
        torch.nn.functional.softmax(x)
    torch.cuda.synchronize()   
    result=(time.time() - start_time) * 1e6 / ITERATION_LOOP_TIME
    return result

@dataclass
class PipeStage():
    instance_name: str = field(default_factory=str)  # 名
    num_acti: int = field(default_factory=int)  # 可存储激活的数量
    num_acti_max: int = field(default_factory=int)
    time_f: float = field(default_factory=float)
    time_b: float = field(default_factory=float)
    f_list: List[int] = field(default_factory=list)  # 接收到的前向batch
    f_time: List[float] = field(default_factory=list)  # 接收到的前向batch到达时间
    b_list: List[int] = field(default_factory=list)  # 接收到的后向batch
    b_time: List[float] = field(default_factory=list)  # 接收到的后向batch到达时间
    status: List[float] = field(default_factory=list)  # 长度为3
    bf_flag:bool=field(default_factory=bool)  # 是否允许反向
    flag: bool = field(default_factory=bool)  # 当前time里是否已行动
    overlap_per: float = field(default_factory=float)

    def __init__(self, stage=[[0.0,0.0]], overlap_per=1.0,
                 acti_form: str = "pp-i",idx=0,  pp=1, vppnum=1):
        self.pp=pp
        self.status = [0.0, 0.0, 0.0]
        self.flag = True
        self.bf_flag= True
        self.num_acti = 0
        self.num_acti_max = eval(acti_form, globals(), {'pp': pp, 'i': idx,'vpp': vppnum})
        self.instance_name = ""
        self.overlap_per = overlap_per
        self.time_f:list = []
        self.time_b:list = []
        self.f_list = []
        self.f_time = []
        self.b_list = []
        self.b_time = []
        self.pipeline_waittime=[]
        if vppnum == len(stage)/pp:
            for i in range(vppnum):
                self.time_f.append(stage[idx+pp*i][0])
                self.time_b.append(stage[idx+pp*i][1])
        else:
            self.time_f.append(stage[idx][0])
            self.time_b.append(stage[idx][1])

    def get_time_f(self,i):
        return self.time_f[int((i/self.pp)% len(self.time_f))]#先看是第几组，再循环索引，最后取值

    def get_time_b(self,i):
        return self.time_b[int((i/self.pp)% len(self.time_b))]#先看是第几组，再循环索引，最后取值

# @dataclass
# class CommStatus(Enum):
#     """通讯状态枚举"""
#     PENDING_REDUCE = auto()  # 待reduce
# @dataclass
# class ScatterPrimitive:
#     """通讯原语结构体 - Scatter操作"""
#     location: str= ""       #通讯发生位置
#     primitive: str = ""       # 通讯原语
#     content: str = ""       # 通讯内容
#     data_volume: str = ""  #通讯数据量
#     output: str = ""     #通讯后输出
#     status: CommStatus = CommStatus.PENDING_REDUCE  # 通讯后状态
@dataclass
class Optimizer:
    """
    优化器类，包含权重、梯度和优化器状态的参数
    """
    weight_size: int = 0  # 权重大小（字节）
    gradient_size: int = 0  # 梯度大小（字节）
    optimizer_state_size: int = 0  # 优化器状态大小（字节）

@dataclass
class OptimizationConfig:
    """
    优化信息类，包含前向/后向影响及计算资源指标
    """
    instance_name: str =field(default_factory=str)  # 实例名
    # 1. 影响的激活值（算子阶段）
    activation_list:List[dict] = field(default_factory=list)
    # 1. 影响的激活值（并行阶段）
    activation_values:List[dict] = field(default_factory=list)
    # 2. 影响的模型参数值（）
    module_params: List[dict]=field(default_factory=list)
    # 3. 影响的权重
    W: List[dict]=field(default_factory=list)
    # 4.影响的算力值：
    compute_power: float =field(default_factory=float)    # 单位FLOPs
    # 5. 前向/后向通讯影响信息（ScatterPrimitive类）
    # forward_impact: ScatterPrimitive=field(default_factory=ScatterPrimitive)
    # backward_impact: ScatterPrimitive=field(default_factory=ScatterPrimitive)
    # 6.影响的并行模块
    parallel_impact:List[dict]=field(default_factory=list)
    #7.操作流图
    flow:List[dict] = field(default_factory=list)

    def __init__(self,instance_name:str=""):
        self.instance_name=instance_name
        self.activation_list=[]
        self.activation_values=[]
        self.module_params=[]
        self.W=[]
        self.flow=[]
        self.compute_power=0.0
        # self.forward_impact=ScatterPrimitive()
        #self.backward_impact=ScatterPrimitive()
        self.parallel_impact=[]
    def set_valuerange(self):
        return [None,[]]

    def apply(self, instance):
        if isinstance(instance,ModuleConfig):#针对1,2,3,4参数
            for weight in self.W:
                if weight['opera'] == "alt" or weight['opera'] == "share":
                    for d in instance.W:
                        if weight["type"] == d['name']:
                            if weight['opera'] == "alt":
                                d['shape'][weight['idx']]=d['shape'][weight['idx']].replace(weight['old_shape'],weight['shape'])
                            else:  
                                d['name']=d['name']+"-share"
                elif weight['opera'] == "add":
                    instance.W.append({"name": weight['name'], "shape": weight['shape']})
                elif weight['opera'] == "del":
                    instance.W = [
                        d for d in instance.W 
                        if d["name"] != weight['name']
                    ]
            for activate in self.activation_list:
                if activate['opera'] == "add":
                    instance.activation_list.append(activate['type'])
                elif activate['opera'] == "del":
                    instance.activation_list = [ d for d in instance.activation_list if d != activate['type']]
            for flow in self.flow:
                if flow['opera'] == "del":
                    instance.flow = [ d for d in instance.flow if d['type'] != flow['type']]
        if isinstance(instance,ParallelConfig):#针对1,2,3,4,6参数
            for impact in self.parallel_impact:
                if impact['opera'] == "add":
                    instance.parallel_impact.append({"name":impact['name']})
                elif impact['opera'] == "del":
                    instance.parallel_impact = [
                        d for d in instance.parallel_impact 
                        if d["name"] != impact['name']
                    ]
                # elif impact['opera'] == "alt":
                #     for d in instance.parallel_impact:
                #         if d["name"] == impact['name']:
                #             d['values']=modify_expr(d['values'],impact['old_values'],impact['values'])
            for weight in self.W:
                if weight['opera'] == "alt":
                    for d in instance.W:
                        if d["name"] == weight['name']:
                            d['shape'][weight['idx']]=d['shape'][weight['idx']].replace(weight['old_shape'],weight['shape'])
                elif weight['opera'] == "add":
                    instance.W.append({"name": weight['name'], "shape": weight['shape']})
                elif weight['opera'] == "del":
                    instance.W = [
                        d for d in instance.W 
                        if d["name"] != weight['name']
                    ]
            for activate in self.activation_values:
                if activate['opera'] == "alt":
                    for d in instance.activation_values:
                        if d["type"] == activate['type']:                          
                            d['values']=d['values'].replace(activate['old_values'],activate['values'])
                elif activate['opera'] == "add":
                    instance.activation_values.insert(activate['locate'],{"type": activate['type'], "values": activate['values'],"parallel":activate['parallel']})
                elif activate['opera'] == "del":
                    instance.activation_values = [ d for d in instance.activation_values if d["type"] != activate['type']]   
        return 0
    
@dataclass
class ModuleConfig:
    """
    模块配置类，包含模块的基本参数和配置选项
    """
    print_flag: bool = False
    use_bias: bool = False  # 是否启用偏置项
    ioput: str ='input'  # 默认等于'input'

    instance_name: str =field(default_factory=str)  # 实例名
    # 1. 模块输入
    input_shape: list=field(default_factory=list)
    
    # 2. 模块输出
    output_shape: list=field(default_factory=list)
       
    # 4. 模型权重
    W:List[dict]=field(default_factory=list)
    
    # 4.5 模型计算流程
    flow:List[dict]=field(default_factory=list)
    #4.7 影响的激活值（名字列表）
    activation_list:List[str] =field(default_factory=list)
    # 5. 影响的激活值（实际内容，按照前向产生的顺序放入）
    activation_values:List[dict] =field(default_factory=list)
    activation_values_auto:List[dict] =field(default_factory=list)
    
    # 6. 影响的模型参数值（按照前向产生的顺序放入）
    module_params: List[dict]=field(default_factory=list)
    module_params_auto:List[dict]=field(default_factory=list)
    # 7. 并行策略通信点
    parallel_impact:List[dict]=field(default_factory=list)

    # 8.流程点(calc_module用)：
    parallel_mode:str=field(default_factory=str)
    # 单位FLOPs
    constant_auto:int=field(default_factory=int)
    time_auto:List[dict]=field(default_factory=list)
    children_auto:list=field(default_factory=list)
    # -------------------- 优化实例 --------------------
    # 实际使用的优化实例
    optimizations: List[OptimizationConfig]=field(default_factory=list)
    #recompute_granularity=None
    #recompute_modules=[]
    recompute_flag:bool = field(default_factory=bool)
    # -------------------- 方法 --------------------
    def __init__(self,args,flag_moe=False,instance_name:str=""):
        self.instance_name=instance_name
        self.args=args
        self.input_shape=[]
        self.output_shape=[]
        #self.ioput='input'
        #self.use_bias=False
        #self.print_flag=False
        self.use_dropout=""
        self.activation_list=[]
        self.activation_values=[]
        self.activation_values_auto=[]
        self.children_auto=[]
        self.W=[]
        self.flow=[]
        self.module_params=[]
        self.module_params_auto=[]
        self.time_auto=[]
        self.constant_auto=0
        self.parallel_impact=[]
        self.parallel_mode=""
        self.compute_power=0.0
        self.optimizations=[]
        self.seq='seq'
        self.hidden='hidden'
        self.h_4='h_4'
        self.h_ffn='h_ffn'
        self.attention_dropout = args.attention_dropout        
        self.hidden_dropout = args.hidden_dropout       # 是否启用dropout
        self.Topk=args.moe_router_topk
        self.flag_moe=flag_moe
        self.recompute_flag=False
        self.recompute()
    
    def recompute(self):
        if self.args.recompute_granularity == 'selective':
            for name in self.args.recompute_modules:
                if (name == 'moe' and self.flag_moe) or (name in self.instance_name and self.flag_moe == False):#False剔除moe层，True确保只是moe层
                    self.recompute_flag=True

    def get_children_auto(self,name:str):
        for idx in self.children_auto:
            if idx.instance_name == name:
                return idx
        return None

    def rename(self,instance_name:str=""):
        self.instance_name=instance_name

    #tp_cut=[right_cut,left_cut].right_cut/left_cut=['head','group']
    def calc_module(self,idx:dict,flag=False,ioput="",acti=1,moe_flag=False,insert=None,tp_cut=[None,None]):
        def interfunc(left2,right2,parallel_style):
            idx['right-matrix']=copy.deepcopy(right2)
            idx['left-matrix']=copy.deepcopy(left2)
            for i in range(len(left2)):
                left2[i]=left2[i].replace(f"/{parallel_style}",'')
            for i in range(len(right2)):
                right2[i]=right2[i].replace(f"/{parallel_style}",'')          
            return left2,right2
        
        if ioput!='': 
            self.activation_list.append(ioput)
        parallel=idx['parallel']
        left:list=copy.deepcopy(idx['left-matrix'])
        right:list=copy.deepcopy(idx['right-matrix'])
        # if idx['type'] == 'matmul':
        #     import pdb; pdb.set_trace() 

        tensor_parallel_style='tp'
        moe_tp_extend_ep=getattr(self.args, 'moe_tp_extend_ep', False)
        moe_extended_tp =getattr(self.args, 'moe_extended_tp', False)
        expert_tensor_parallel_size =getattr(self.args, 'expert_tensor_parallel_size', None)
        if expert_tensor_parallel_size is None:
            expert_tensor_parallel_size =getattr(self.args, 'expert_model_parallel_size', None)    
        #启用了这个，就不会切expert了，全部切expert里的权重，这是为了负载均衡的考虑
        if self.flag_moe and 'ffn' in self.instance_name:
            if moe_extended_tp or expert_tensor_parallel_size is not None :
                tensor_parallel_style='tp/ep'
            if moe_tp_extend_ep:
                tensor_parallel_style='1'

        if moe_flag:
            idx['type']=f"moe-{idx['type']}"

        #ep不在这里切，因为还有共享专家的存在,共享专家一切不变，ep的话每次临时切分
        #而且EP属于DP的变种，也不参与总节点限制
        # if 'ep' in parallel:#目前固定放第一个
        #     if 'epf' not in parallel:
        #         if 'capacity' in left[-2]:
        #             left[-2]=f'seq'
        #         else:
        #             left[-2]=f"seq*(capacity*Topk)/route_experts"
                #left[-2]='seq'#极端情况
                #left[-2]='seq*Topk/route_experts'#理想情况
        # else:
        #     left,right=interfunc(left,right,'ep')
        #     if 'ep'  not in tensor_parallel_style:
        #         left[-2]=f"{left[-2]}/ep"#变成dp

        left,right=interfunc(left,right,'cp')#这里解决了[s,s]的问题，但是right的形状，对于cp通信来说不对，去cp通信里处理
        left[-2]=f"{left[-2]}/cp"

        left,right=interfunc(left,right,'ulyp')

        if 'ulyp' in parallel :#ulyp and ulypf
            if 'matmul-Wqkv' in idx['type']:
                left[-2]=f"{left[-2]}/ulyp"
            else:#ulypf and matmul-o
                left=copy.deepcopy(idx['left-matrix'])
                right=copy.deepcopy(idx['right-matrix'])  

        if 'ulyp' not in parallel :
            left[-2]=f"{left[-2]}/ulyp"
                   
        left,right=interfunc(left,right,'tp')
        left,right=interfunc(left,right,'ep')
        tp_flag=False
        sp_flag=True
        #这个是反过来了，把tp挪到ep了。
        #先不做，看到再说吧
        # if self.flag_moe and 'ffn' in self.instance_name and self.args.moe_tp_extend_ep:
        #     sp_flag=False
        #     parallel=parallel.replace('tpf','')
        #     parallel=parallel.replace('tp','')
        #     left=copy.deepcopy(idx['left-matrix'])
        #     right=copy.deepcopy(idx['right-matrix'])

        if 'tp' in parallel and 'tpf' not in parallel:#tp
            #确认是哪个权重参数的
            for weight in self.W:
                if weight['name'] in idx['type']:
                    idx['param']=weight['shape']
            tp_flag=True
            if "col" in self.parallel_mode:
                if tp_cut[0] is None:
                    right[-1]= f"{right[-1]}/{tensor_parallel_style}"
                else:
                    for name in tp_cut[0]:
                        right[-1]= right[-1].replace(f"{name}",f"{name}/{tensor_parallel_style}")
            else: 
                if tp_cut[0] is None or tp_cut[1] is None:
                    right[-2]= f"{right[-2]}/{tensor_parallel_style}"
                    left[-1] = f"{left[-1]}/{tensor_parallel_style}"
                else:
                    for name in tp_cut[0]:
                        right[-2]= right[-2].replace(f"{name}",f"{name}/{tensor_parallel_style}")
                    for name in tp_cut[1]:
                        left[-1] = left[-1].replace(f"{name}",f"{name}/{tensor_parallel_style}")

        elif 'tpf' in parallel:#tpf
            left=copy.deepcopy(idx['left-matrix'])
            right=copy.deepcopy(idx['right-matrix'])

        left,right=interfunc(left,right,'sp')

        if 'tp' not in parallel and sp_flag: 
            left[-2]=f"{left[-2]}/sp"
            # if 'ep' in tensor_parallel_style and 'ep' not in left[-2]:
            #     left[-2]=f"{left[-2]}/ep"
        
        for w in self.W:
            if w['name'] in idx['type']:
                weight=copy.deepcopy(right[-len(w['shape']):])
                if 'matmul-Wqkv' in idx['type']:
                    weight[-1]=weight[-1].replace('head','head/ulyp')
                    weight[-1]=weight[-1].replace('group','group/ulyp')
                w['shape']=weight[-len(w['shape']):]
        
        if 'sub' in idx['type'] or 'add' in idx['type']:
            right=left

        idx['left-matrix']=left
        idx['right-matrix']=right
        output=copy.deepcopy(left)

        if tp_flag and 'matmul' in idx['type'] and 'tp' not in right[-1]:
            output[-2]=f"{output[-2]}/sp"
        
        self.seq=output[-2]
        if 'matmul-Wqkv' in idx['type']:
            self.seq=self.seq.replace('/ulyp','')

        if "layernorm" in idx["type"]:
            idx['output-matrix']=copy.deepcopy(output)
            output.append(f"8")
            idx['calc']="*".join(output)
        elif "rmsnorm" in idx["type"]:
            idx['output-matrix']=copy.deepcopy(output)
            output.append(f"5")
            idx['calc']="*".join(output)
        elif "softmax" in idx["type"]:
            idx['output-matrix']=copy.deepcopy(output)
            output.append(f"17")
            if 'attention' in self.instance_name and self.args.attention_softmax_in_fp32:
                output.append('2')
            idx['calc']="*".join(output)
        elif "matmul" in idx["type"]:
            output[-1]=right[-1]
            idx['output-matrix']=copy.deepcopy(output)
            output.append(f"2 * {left[-1]}")
            idx['calc']="*".join(output)
        elif "index" in idx["type"]:
            idx['calc']="0"           
        elif "experts" in idx["type"]:
            idx['output-matrix']=copy.deepcopy(output)
            idx['calc']=f"(2*Topk+1)*b*{self.seq}*{self.hidden}" 
        else:#mul
            idx['output-matrix']=copy.deepcopy(output)
            idx['calc']="*".join(output)

        if flag:
            for _ in range(acti):
                self.activation_list.append(idx['type'])
        if int == type(insert):
            self.flow.insert(insert,idx)
        else:
            self.flow.append(idx)
        return 
    
    def calculate_flow(self,dtype=False):
        self.activation_values=[]
        self.module_params=[]
        self.parallel_impact=[]
        self.input_shape[-2]=f'{self.input_shape[-2]}/sp/cp/ulyp'
        input_value='*'.join(self.input_shape)
        if dtype:
            self.activation_values.append({"type": 'self_dtype', "values":f'2*{input_value}*sp/head',"parallel":''})#其实是4*fb8，这里改成2*fb16
        if 'input' in self.activation_list:
            self.activation_values.append({"type": 'input', "values":input_value,"parallel":self.flow[0]['parallel']})
        if 'output' in self.activation_list:
            self.activation_values.append({"type": 'output', "values":'*'.join(self.output_shape),"parallel":self.flow[-1]['parallel']})
        expert_parallel_style='ep'
        moe_tp_extend_ep=getattr(self.args, 'moe_tp_extend_ep', False)
        moe_extended_tp =getattr(self.args, 'moe_extended_tp', False)
        expert_tensor_parallel_size =getattr(self.args, 'expert_tensor_parallel_size', None)
        if expert_tensor_parallel_size is None:
            expert_tensor_parallel_size =getattr(self.args, 'expert_model_parallel_size', None) 
        if self.flag_moe and 'ffn' in self.instance_name:
            if moe_extended_tp or expert_tensor_parallel_size is not None :
                expert_parallel_style='1'
            if moe_tp_extend_ep:
                expert_parallel_style='tp*ep'
        #参数
        #这是为了不让门控网络也加入moe的倍增，所以单独拿出来
        if 'Network' in self.W[-1]['name']:
            self.module_params.append({"type": f"param-{self.W[-1]['name']}", "values": '*'.join(self.W[-1]['shape'])})#不考虑偏置
        for w in self.W:
            values='0'
            if 'Network' in w['name'] or '-share' in w['name']:#针对Moe前的网络层/output层的共享操作
                values='0'
            else:
                values='*'.join(w['shape'])
                #if self.args.skip_bias_add == False and self.use_bias:
                if self.use_bias:
                    values=f"({values}+{w['shape'][-1]})"
                if self.flag_moe and "ffn" in self.instance_name:
                    values=f'h_share_moe*{values}/h_moe'
                    expert_params=f'(route_experts/{expert_parallel_style})*{values}'
                    self.module_params.append({"type": f"moe_route-param-{w['name']}", "values": f"{expert_params}"})
                    #values=f"(route_experts/{expert_parallel_style})*{values}+(h_share_moe*{values}/h_moe)"
            self.module_params.append({"type": f"param-{w['name']}", "values": f"{values}"})
        #激活
        for obj in self.flow:
            values='0'
            if obj['type'] in self.activation_list:#激活
                values='*'.join(obj['output-matrix'])
                if self.activation_list.count(obj['type']) > 1:
                    values=f"{self.activation_list.count(obj['type'])}*{values}"
                if 'moe' in obj['type']:#moe_flag
                    if 'index-comm' in obj['type']:#在本家算子开始之前，会进行expert的permute，这里有个通算覆盖点
                        values=f'capacity*Topk*{values}'
                    elif 'index-experts' in obj['type']:
                        values=f'(Topk+1)*{values}'#Topk是因为各专家token回来以后，肯定是topk个s,然后还一个本家的s。每个s都有[b,s,h]
                    else:#个数*大小+本家大小
                        values=f'(capacity*Topk*{values})+(h_share_moe*{values}/h_moe)'
                self.activation_values.append({"type": obj['type'], "values":f"{values}","parallel":obj['parallel']})
            if obj['parallel'] != "" or obj['calc'] !="0":#通讯计算放一起，因为共享同步点
                if "tpf" not in obj['parallel'] and "tp" in obj['parallel']:
                    self.parallel_impact.append({"type":f"{obj['type']}","to_memory":f"{values}","right-matrix":obj['right-matrix'],"left-matrix":obj['left-matrix'],"values":obj['output-matrix'],"calc":obj['calc'],"parallel":obj['parallel'].replace(',', '*'),"parallel_mode":obj['parallel_mode'],'param':obj['param']})     
                else:
                    self.parallel_impact.append({"type":f"{obj['type']}","to_memory":f"{values}","right-matrix":obj['right-matrix'],"left-matrix":obj['left-matrix'],"values":obj['output-matrix'],"calc":obj['calc'],"parallel":obj['parallel'].replace(',', '*')})
        return      

    def get_module_params(self) -> str:
        return self.cal_params()
    
    def cal_params(self,module_params=[]) -> str:
        resultlist=[]
        resultlist_m=[]
        if module_params==[]:
            module_params=self.module_params 
        for params in module_params:
            if params['values'] == "": continue
            if 'moe_route' in params['type']:
                resultlist_m.append(f"({params['values']})")
            else:
                resultlist.append(f"({params['values']})")
        return f"({'+'.join(resultlist)})",f"({'+'.join(resultlist_m)})"
    
    def get_activation(self) -> str:
        activation_values_list=copy.deepcopy(self.activation_values)
        result,result_m=self.cal_activation(activation_values_list)
        if result == "()":
            return "0"
        elif result_m=="()":
            return f"({SIZE_BF16} * {result})"   
        else:return f"({SIZE_BF16} * {result} + {result_m})" 
    
    def cal_activation(self,activation_values=[]) -> tuple[str, str]:
        if self.recompute_flag:
            return "(b*seq*hidden)","()"#只存一个输入，其余全部重算
        resultlist=[]
        resultlist_m=[]
        temp=""
        if activation_values==[]:
            activation_values=self.activation_values
        for activation in activation_values:
            if activation['values']=="": continue
            temp=f"{activation['values']}"
       
            if "mask" in activation['type'] or "dropout" in activation['type']:
                resultlist_m.append(temp)
            else:
                resultlist.append(temp)
        return f"({'+'.join(resultlist)})",f"({'+'.join(resultlist_m)})" 

    def get_flow_nodes(self,ModelConfig) -> str:
        """
        返回模块激活值的计算结果（由子类实现）
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def choose_optimizations(self,optimizations: List[OptimizationConfig]):
        for opt in optimizations:
            self.optimizations.append(opt)
    @staticmethod
    def apply_basic_optimizations(opt: List[OptimizationConfig],print_flag=False):
        """
        应用所有启用的优化类
        """
         
        opt.apply(ModuleConfig)
        if print_flag:
            print(f"basic_optimization finish: {opt.instance_name}")


    def apply_optimizations(self,optimizations: List[OptimizationConfig]):
        """
        应用所有启用的优化类
        """
        for opt in optimizations:
            # 调用优化类的apply_optimization方法
            opt.apply(self)
            self.instance_name=f"{self.instance_name}-{opt.instance_name}"
            if self.print_flag:
                print(f"optimization finish: {opt.instance_name}")
@dataclass
class ParallelConfig:
    """
    并行配置基类，用于管理并行计算相关的配置和影响信息
    """
    param_all_comm=('0')
    instance_name:str=field(default_factory=str)
    # -------------------- 前向/后向影响信息 --------------------
    # forward_impact: ScatterPrimitive = field(default_factory=ScatterPrimitive)  # 前向传播影响
    # backward_impact: ScatterPrimitive = field(default_factory=ScatterPrimitive)  # 后向传播影响

    # -------------------- 通讯时最低带宽 --------------------
    min_bandwidth: float = field(default_factory=float)  # 单位MB/s,采用真实架构时有用

       
    # 5. 影响的激活值（按照前向产生的顺序放入）
    activation_values:List[dict] =field(default_factory=list)
    
    W:List[dict]=field(default_factory=list)
    # 6. 影响的模型参数值（按照前向产生的顺序放入）
    module_params: List[dict]=field(default_factory=list)
    
    # 4. 并行策略作用域（优化器、模块、融合算子。如果是模块的话，继续去调用模块的方法，让模块决定自己被如何并行；如果是融合算子的话，默认进算子方法，让算子决定自己能不能被并行）
    parallel_impact:List[dict]=field(default_factory=list)

    # 5.影响的算力值：
    compute_power: float =field(default_factory=float)    # 单位FLOPs
    # -------------------- 优化实例 --------------------
    # 实际使用的优化实例
    optimizations: List[OptimizationConfig]=field(default_factory=list)
    print_flag: bool = field(default_factory=bool)
    def __init__(self,args,instance_name:str="",print_flag=False):
        self.instance_name=instance_name
        self.print_flag=print_flag
        # self.forward_impact=ScatterPrimitive()
        #self.backward_impact=ScatterPrimitive()
        self.min_bandwidth=0.0
        self.activation_values=[]
        self.W=[]
        self.module_params=[]
        self.parallel_impact=[]
        self.compute_power=0.0
        self.optimizations=[]
        self.args=args
        self.layer:int=args.num_layers

    def flowgraph(self):

        return
    
    def apply_optimizations(self):
        """
        应用所有启用的优化类
        """
        for opt in self.optimizations:
            # 调用优化类的apply_optimization方法
            opt.apply(self)
            if self.print_flag:
                print(f"optimization finish: {opt.instance_name}")    

@dataclass
class Data_parallel(ParallelConfig):
    """
    Data_parallel类，继承自ParallelConfig
    """
    def __init__(self, args,instance_name: str = "dp",optimizations:List=None):
        """
        """
        # 调用父类初始化
        super().__init__(args,instance_name=instance_name)
        
        # 设置通讯带宽为default
        self.min_bandwidth = "default"      

        # 并行时影响的模块（None） 

        #是否启用优化
        if optimizations != None:
            self.optimizations=optimizations
            self.apply_optimizations()
   

    def memory_apply(self,memory_optimizer:str,moe_route_optimizer:str):
        result=memory_optimizer
        result_m=moe_route_optimizer
        for part in self.parallel_impact:
            result=memory_optimizer.replace(part["name"],f"({part['name']}/dp)")
            result_m=moe_route_optimizer.replace(part["name"],f"({part['name']}/dp)")
        return result,result_m
    
    def DP_flowgraph(self,costmodel:List[dict],d_num):
       #dp只有后向的参数同步，（all_reduce)
       #目前dp默认是梯度累积更新，每个epoch更新一次。每次的传输次数为1，数据量为dp*{com_data}(来自深度学习的分布式训练与集合通信（一）)
       #因此直接取消dp的梯度传输，不给dp施加惩罚项，因为dp永远是资源充足的第一选择
        # item=costmodel[0]#最开始的input
        # item["parallel"]="dp"#反向时，input=output，input之后，将梯度与别的dp组共享。
        # #param_list=ParallelConfig.param_all_comm if ParallelConfig.param_all_comm is not None else ['0']
        # com_data='+'.join(param_list)
        # item["primitive"].append(('' , "all_reduce"))
        # item["com_data"].append(('0',f"(dp*({com_data})/tp/pp"))
        return
@dataclass
class Pipe_parallel(ParallelConfig):
    """
    Pipe_parallel类，继承自ParallelConfig
    实现管道并行相关的配置和影响信息
    """
   
    def __init__(self,args, instance_name: str = "pp",optimizations:List=None,acti_form:str = [d["acti_form"] for d in PP_Form if "1F1B" == d["name"] ][0]):
        """
        初始化Pipe_parallel模块
        """
        # 调用父类初始化
        super().__init__(args,instance_name=instance_name)
        
        # 5. 影响的激活值（按照前向产生的顺序放入）
        self.acti_num=acti_form
        
        W:List[dict]=[]
        # 6. 影响的模型参数值（按照前向产生的顺序放入）
        #self.module_params: List[dict]=[]
        # 设置通讯带宽为default
        self.min_bandwidth = "default"
        
        # 并行时影响的模块（None）
        #self.parallel_impact.append({"name":"layer"}) ,已废弃，不用Layer来表示
        # 可选优化类(会对Pipe本身产生影响，并行对内存模型的影响，将会在各个并行模块的优化类遍历完成之后，统一再作用到内存模型上)
        #是否启用优化
        if optimizations != None:
            self.optimizations=optimizations
            self.apply_optimizations() 
    def PP_flowgraph(self,costmodel:List[dict],p_num):
        #前向+后向
        cmdp=[d for d in costmodel if ("input" in d["name"] or"output" in d["name"] )]
        for opti in self.optimizations:
            if 'VirtualPipe' == opti.instance_name and opti.get_vppnum(p_num) != 1:
                p_num=p_num*opti.get_vppnum(p_num)#真实切分=pp*vpp
        listdp=np.array_split(cmdp,p_num) 
        for i in range(len(listdp)):
            item=listdp[i]
            item[0]["parallel"]="pp"
            item[-1]["parallel"]="pp"
            if i == 0:    
                item[0]["primitive"].append(("",""))
                item[0]["com_data"].append(("0","0"))
            else:
                item[0]["primitive"].append(("recv","send"))#后向算完梯度后，继续传递梯度和DP维度下同步梯度是两个独立的事，可以一起做。通常情况all_reduce肯定比send慢
                item[0]["com_data"].append(("b*seq*hidden","b*seq*hidden"))  
            if i == len(listdp)-1:   
                item[-1]["primitive"].append(("",""))
                item[-1]["com_data"].append(("0","0"))
            else:
                item[-1]["primitive"].append(("send","recv"))
                item[-1]["com_data"].append(("b*seq*hidden","b*seq*hidden"))
        return
        #pipe在参数部分，有前，中，后的不同，在激活部分，第一个承受的中间激活最多
    def calc_actinum(self,pp):
        numlist:List[float]=[]
        vppnum=1
        for opti in self.optimizations:
                if 'VirtualPipe' == opti.instance_name:
                    vppnum=opti.get_vppnum(pp)
        for i in range(pp):
            num:float=eval(self.acti_num, globals(), {'pp': pp, 'i': i})
            if vppnum > 1:
                num:float=eval(self.acti_num, globals(), {'pp': pp, 'i': 0})
                num=num*(float((pp * vppnum + pp-1 -2*i )/(pp * vppnum)))+ 0.27#0.27是测出来的启发式激活#实际上b会被vpp切成vpp份，所以这里采用float来简化s/vpp
            numlist.append(num)
        return numlist,vppnum
    
    def memory_apply(self,pp:int,memory_param:List[str],memory_activation:List[str],moe_route_param:List[str]):
        recompute_value=None
        recompute_method=None
        recompute_num_layers=1
        if self.args.recompute_granularity == 'full':
            recompute_method=self.args.recompute_method if self.args.recompute_method is not None else 'uniform' #'uniform' or 'block'
            recompute_num_layers=self.args.recompute_num_layers if self.args.recompute_num_layers is not None else 1 # int 
        cut=self.layer/pp
        #import pdb; pdb.set_trace() 
        mm_param_list=memory_param[1].split('&')[:int(cut)]#取单节点物理层面上的参数和激活
        mm_acti_list=memory_activation[1].split('&')[:int(cut)]#
        mm_moe_param_list=moe_route_param[0].split('&')[:int(cut)]
        mm_moe_param='+'.join(mm_moe_param_list)
        mm_param='+'.join(mm_param_list)
        mm_acti=f"({'+'.join(mm_acti_list)})"
        pp_param=[]
        pp_acti=[]
        pp_moe_route_param=[]
        if pp > 1:  
            numlist,vppnum=self.calc_actinum(pp)
            virtual_layers=self.layer/(vppnum*pp)
            if recompute_method=='uniform':#无限次生效，隔几层存一次输入
                uniform_num=math.ceil(virtual_layers/recompute_num_layers)#3层存一次,那么4,5,6层都是存2次。
                recompute_value=f'{uniform_num}*{vppnum}*b*seq/sp/cp/ulyp*hidden'
            if recompute_method=='block':#有限次生效，只有前几层不存，单存最开始的输入+后几层的全部激活
                block_num:float=max(0,virtual_layers-recompute_num_layers)/virtual_layers
                recompute_value=f'(b*seq/sp/cp/ulyp*hidden+{block_num}*{mm_acti})'#最开始需要存一份检查点
            if recompute_value:
                mm_acti=recompute_value
            for i in range(len(numlist)):
                num=numlist[i]
                if i == 0:
                    pp_moe_route_param.append(['0',mm_moe_param])
                    pp_param.append([memory_param[0],mm_param])
                    pp_acti.append([f"{num}*{memory_activation[0]}",f"{num}*{mm_acti}"])
                elif i == pp-1:
                    pp_moe_route_param.append([mm_moe_param,'0'])
                    pp_param.append([mm_param,memory_param[2]])
                    pp_acti.append([f"{num}*{mm_acti}",f"{num}*{memory_activation[2]}"])
                else:
                    pp_moe_route_param.append([mm_moe_param])
                    pp_param.append([mm_param])
                    pp_acti.append([f"{num}*{mm_acti}"])
        else:
            if recompute_method=='uniform':#无限次生效，隔几层存一次输入
                uniform_num=math.ceil(self.layer/recompute_num_layers)#3层存一次,那么4,5,6层都是存2次。
                recompute_value=f'(2*{uniform_num}*b*seq/sp/cp/ulyp*hidden)'
            if recompute_method=='block':#有限次生效，只有前几层不存，单存最开始的输入+后几层的全部激活
                block_num:float=max(0,self.layer-recompute_num_layers)/self.layer
                recompute_value=f'(2*(b*seq/sp/cp/ulyp*hidden+{block_num}*{mm_acti}))'#最开始需要存一份检查点
            if recompute_value:
                mm_acti=recompute_value
            mm_param=f'{memory_param[0]}+{mm_param}+{memory_param[2]}'
            mm_acti=f'{memory_activation[0]}+{mm_acti}+{memory_activation[2]}'
            pp_param.append([mm_param])
            pp_acti.append([mm_acti])
            pp_moe_route_param.append([mm_moe_param])

        return pp_param,pp_acti,pp_moe_route_param

@dataclass
class Context_parallel(ParallelConfig):
    """
    MindSpeed的Ring Attention
    开启Context Parallel时需要同时开启Flash Attention特性，否则特性不支持。
    在使用GPT类模型进行训练的场景下，建议attention-mask-type设置为causal。
    在8k的序列长度情况下，由于计算的时间缩短，cp功能分割之后的send receive的时间反而会长于计算时间，造成性能的下降，所以建议配置seq-length / context-parallel-size> 8k以获取最佳效果。具体公式参考：S/(Talpha) >= 1/(Wbeta)，其中，S=seq-length / context-parallel-size， T表示芯片的理论算力，alpha表示计算效率，W表示理论通信带宽，beta表示带宽利用率。
    内层窗口--cp-window-size增大时，通信与计算并发程度更高，但是计算、通信并发时可能由于片上内存带宽抢占，整体效率下降，需要结合实际场景进行调试，例如llama2裁剪模型32k序列长度，cp为16且无其他并行切分时，实测内层窗口大小为2时性能最优。
    """
    def __init__(self, args,instance_name: str = "cp",optimizations:List=None):
        """
        """
        # 调用父类初始化
        super().__init__(args,instance_name=instance_name)
        
        
        # 设置通讯带宽为default
        self.min_bandwidth = "default"
              
        # 并行时影响的模块（None）
        self.memory_modules = ""
        
        #是否启用优化
        if optimizations != None:
            self.optimizations=optimizations
            self.apply_optimizations() 

    def CP_flowgraph(self,costmodel:List[dict],c_num):
        #前向+后向
        cmcu=[d for d in costmodel if "cp" in d["parallel"] and "cpf" not in d["parallel"] ]
        for item in cmcu:
            if "score" in item["name"]:
                    item["primitive"].append(("all_gather","reduce_scatter"))
                    item["com_data"].append((f"2*(cp - 1)*({'*'.join(item['right-matrix'])})",f"2*(cp - 1)*({'*'.join(item['right-matrix'])})"))#底层实现的send和recv没有重叠，所以乘2
            if "matmul-o" in item["name"]:
                item["primitive"].append(("all_gather","reduce_scatter"))
                item["com_data"].append((f"2*(cp - 1)*({'*'.join(item['right-matrix'])})",f"2*(cp - 1)*({'*'.join(item['right-matrix'])})"))
        return
@dataclass
class Ulyssess_parallel(ParallelConfig):
    """
    Data_parallel类，继承自ParallelConfig
    """
    def __init__(self, args,instance_name: str = "ulyp",optimizations:List=None):
        """
        """
        # 调用父类初始化
        super().__init__(args,instance_name=instance_name)
        
        # 设置通讯带宽为default
        self.min_bandwidth = "default"
        
        # 其他并行对该模块的影响（None）
        self.other_parallel_impact = None
        
        # 并行时影响的模块（None）
        self.memory_modules = None
        
        #是否启用优化
        if optimizations != None:
            self.optimizations=optimizations
            self.apply_optimizations() 

    def UP_flowgraph(self,costmodel:List[dict],u_num):
        #前向+后向
        cmcu=[d for d in costmodel if"ulyp" in d["parallel"] and "ulypf" not in d["parallel"] ]
        for item in cmcu:
            #从[b,head,seq/ulyp,c]变成[b,head/ulyp,seq,c]。通讯模块大小为[b,head/ulyp,seq/ulyp,c]。因此通讯为(2*ulyp)*('*'.join([b,head,seq/ulyp,c])/ulyp)=2 * ('*'.join([b,head,seq/ulyp,c]))
            if "Wqkv" in item["name"]:
                    item["primitive"].append(("alltoall","alltoall"))
                    item["com_data"].append((f"2 *({'*'.join(item['values'])})",f"2 * ({'*'.join(item['values'])})"))
                    item["comm_output"]=len(item["primitive"])-1
            if "matmul-o" in item["name"]:
                item["primitive"].append(("alltoall","alltoall"))
                item["com_data"].append((f"2*({'*'.join(item['values'])})",f"2*({'*'.join(item['values'])})"))#
                item["comm_output"]=len(item["primitive"])-1
        return
    
@dataclass
class Tensor_parallel(ParallelConfig):
    """
    Data_parallel类，继承自ParallelConfig
    """
    def __init__(self, args,instance_name: str = "tp",optimizations:List=None):
        """
        """
        # 调用父类初始化
        super().__init__(args,instance_name=instance_name)
        #sp和tp切分的维度不同，所以sp和tp互转时，存在先后通讯。先sp 再计算，在tp
        #前向，后向，次数
        #张量并行下的通信原语和对应次数，都来自深度学习的分布式训练与集合通信（二）
        self.reshape_mapping = {'col-col': ['all_gather','reduce_scatter','(tp-1)','(tp-1)','tp'], 
                           'row-row': ['reduce_scatter','all_gather','(tp-1)','(tp-1)','tp'], 
                           'row-col': ['all_reduce','all_reduce','(tp-1)','(tp-1)','tp'], 
                           'col-row': ['','','0','0',''],
                           'col-':['reduce_scatter','reduce_scatter','(tp-1)','(tp-1)','sp,tp'],#块的大小为b*seq/sp*h/tp
                           '-col':['all_gather','reduce_scatter','(tp-1)','(tp-1)','sp'],
                           'row-':['reduce_scatter','all_gather','(tp-1)','(tp-1)','sp'],
                           '-row':['reduce_scatter','reduce_scatter','(tp-1)','(tp-1)','sp,tp']#块的大小为b*seq/sp*h/tp                                      
                            }   
        # 设置通讯带宽为default
        self.min_bandwidth = "default"   
        # 其他并行对该模块的影响（None）
        self.other_parallel_impact = None 
        # 并行时影响的模块（None）
        self.memory_modules = None
        #是否启用优化
        if optimizations != None:
            self.optimizations=optimizations
            self.apply_optimizations()
    

    # cmts=[d for d in costmodel if ("tp" in d["parallel"] and "tpf" not in d["parallel"])]
        # for i in range(cmts.__len__()-1):
        #     com_data=''
        #     temp=[]
        #     if 
        #     if cmts[i]["num"] != cmts[i+1]["num"] :#不在同一个算子层里，中间经过了sp转化
        #             result=self.reshape_mapping.get(f"{cmts[i]['parallel_mode']}-" , "输入无效") 
        #     else:#在同一个算子层，判断通讯
        #         if "sp" in cmts[i+1]["parallel"] and "spf" not in cmts[i+1]["parallel"]: 
        #             if s_num == 1: result=self.reshape_mapping.get(f"{cmts[i]['parallel_mode']}-", "输入无效")
        #             else:          result=self.reshape_mapping.get(f"{cmts[i]['parallel_mode']}-sp", "输入无效")
        #         else:              result=self.reshape_mapping.get(f"{cmts[i]['parallel_mode']}-{cmts[i+1]['parallel_mode']}", "输入无效")

    def TP_SP_flowgraph(self,costmodel:List[dict],t_num,s_num):
        #前向+后向
        parallel_mode=""
        for idx in costmodel:
            if "tp" not in idx["parallel"]:#''
                if parallel_mode == "":
                    continue
                else:
                    result=self.reshape_mapping.get(f"{parallel_mode}-" , "输入无效")
                    parallel_mode=""
            elif ("tp" in idx["parallel"] and "tpf" not in idx["parallel"]):#'tp'
                result=self.reshape_mapping.get(f"{parallel_mode}-{idx['parallel_mode']}" , "输入无效")
                parallel_mode=idx['parallel_mode']
            else:#'tpf'
                continue
                #output_matrix=idx['values']
            temp=copy.deepcopy(result)
            com_data='*'.join(idx['left-matrix'])
            com_data=com_data.replace('/tp','').replace('/sp','')
            if 'tp' in temp[4]:#做切分
                com_data=f"{com_data}/tp"
            if 'sp' in temp[4]:
                com_data=f"{com_data}/sp"
            #不同通信之间，是竞争关系，取通信耗时最长的那组为整个通信耗时。
            # （为什么可以并发，因为任意两点只属于一个并行组合，并行组合并发通信，并不会导致两点之间出现通信内容不一致的情况）
            if self.args.sequence_parallel and 'sp' in temp[4] and 'all_gather' in temp[1]:#针对'row-'的sp通信分离
                temp[0]=f'{temp[0]}, '
                temp[1]=f'{temp[1]},all_gather'#sp的聚合
                idx["com_data"].append((f"({temp[2]}*{com_data}),0",f"({temp[3]}*{com_data}),({temp[3]}*{com_data})")) 
            else:
                idx["com_data"].append((f"({temp[2]}*{com_data})",f"({temp[3]}*{com_data})")) 
            #idx["com_data"].append((f"({temp[2]}*{com_data})",f"({temp[3]}*{com_data})"))
            idx["primitive"].append((temp[0],temp[1]))     
        return
    
@dataclass
class Expert_parallel(ParallelConfig):#SP是TP的优化
    """
    Expert_parallel类，继承自ParallelConfig
    """
    def __init__(self, args,instance_name: str = "ep",optimizations:List=None):
        """
        """
        # 调用父类初始化
        super().__init__(args,instance_name=instance_name)
        #前向，后向，次数 
        # 设置通讯带宽为default
        self.min_bandwidth = "default"
        
        # 其他并行对该模块的影响（None）
        self.other_parallel_impact = None
        
        # 并行时影响的模块（None）
        self.memory_modules = None
        
        #是否启用优化
        if optimizations != None:
            self.optimizations=optimizations
            self.apply_optimizations()
         
    def EP_flowgraph(self,costmodel:List[dict],e_num):
        #前向+后向
        #张量并行下的通信原语和对应次数，都来自深度学习的分布式训练与集合通信（二）
        expert_parallel_style='ep'
        moe_tp_extend_ep=getattr(self.args, 'moe_tp_extend_ep', False)
        moe_extended_tp =getattr(self.args, 'moe_extended_tp', False)
        expert_tensor_parallel_size =getattr(self.args, 'expert_tensor_parallel_size', None)
        if moe_extended_tp or expert_tensor_parallel_size is not None :
            expert_parallel_style='1'
        if moe_tp_extend_ep:
            expert_parallel_style='tp*ep'        
        cme=[d for d in costmodel if "ep" in d["parallel"]]
        com_data=''
        for i in range(cme.__len__()-1):
            com_data='*'.join(cme[i]['values'])
            if 'softmax-all_expert' in cme[i]['name']:
                cme[i]["primitive"].append(("all_gather",""))
                com_dataf=f"({expert_parallel_style}-1)*{com_data}"
                com_datab=f"0"
            if self.args.moe_token_dispatcher_type =='alltoall':
                cme[i]["primitive"].append(("alltoall","alltoall"))
                com_dataf=f"2*capacity*Topk*{expert_parallel_style}*{com_data}"
                com_datab=f"2*capacity*Topk*{expert_parallel_style}*{com_data}"#route_experts约掉了
            else :#allgather
                cme[i]["primitive"].append(("all_gather","all_gather"))
                com_dataf=f"(route_experts-1)*capacity*Topk*{expert_parallel_style}*{com_data}/route_experts"
                com_datab=f"(route_experts-1)*capacity*Topk*{expert_parallel_style}*{com_data}/route_experts"
                
            cme[i]["com_data"].append((com_data,com_data))  
        return
@dataclass
class Included_Modules:
    front:List[ModuleConfig]= field(default_factory=list)
    layer:List[ModuleConfig]= field(default_factory=list)
    middle:List[List[ModuleConfig]]=field(default_factory=list)
    back:List[ModuleConfig]=field(default_factory=list)
@dataclass
class Included_Parallels:
    data_p:Data_parallel=None
    pipe_p:Pipe_parallel=None
    tensor_p:Tensor_parallel=None
    context_p:Context_parallel=None
    ulyssess_p:Ulyssess_parallel=None
    expert_p:Expert_parallel=None
        
    def DataParallel(self,args,optimizations:List=None):
        instance_name=f"dp"
        self.data_p=Data_parallel(args,instance_name,optimizations)

    def PipeParallel(self,args,optimizations:List=None):
        instance_name=f"pp"
        self.pipe_p=Pipe_parallel(args,instance_name,optimizations)
    def TensorParallel(self,args,optimizations:List=None):
        instance_name=f"tp"
        self.tensor_p=Tensor_parallel(args,instance_name,optimizations)

    def ContextParallel(self,args,optimizations:List=None):
        instance_name=f"cp"
        self.context_p=Context_parallel(args,instance_name,optimizations)

    def UlyssessParallel(self,args,optimizations:List=None):
        instance_name=f"ulyp"
        self.ulyssess_p=Ulyssess_parallel(args,instance_name,optimizations)

    def ExpertParallel(self,args,optimizations:List=None):
        instance_name=f"ep"
        self.expert_p=Expert_parallel(args,instance_name,optimizations)

@dataclass
class ModelConfig:
    """
    内存模型基类，用于管理模型配置和计算相关指标
    """
    # -------------------- 模型配置参数 --------------------
    included_modules:Included_Modules=field(default_factory=Included_Modules)
      # 当前模型启用（推荐）的模块组合，包含的模块，分为前端、中间和后端三个部分
    included_modules_list:List[Included_Modules]=field(default_factory=set[Included_Modules])
    #当前模型包含的所有模块组合(启用search_fused_calculator_flag、search_module_flag的情况下)
    included_parallels:Included_Parallels=field(default_factory=Included_Parallels)
     # 当前模型启用（推荐）的并行组合
    included_parallels_list:List[Included_Parallels]=field(default_factory=set[Included_Parallels])
    #当前模型包含的所有并行组合(启用search_parallel_flag的情况下)
    optimizer_options={#添加优化器的地方
        "Adam": Optimizer(weight_size=4, gradient_size=2, optimizer_state_size=12),#这里的optimizer_state_size多了一个gradientFP32
        "Adam-reuse_fp32_param": Optimizer(weight_size=2, gradient_size=2, optimizer_state_size=12)
    }
    moe_route_optimizer="(weight_size + gradient_size + optimizer_state_size)"
    memory_optimizer=moe_route_optimizer
    memory_param_list:List[List[str]]=field(default_factory=list)
    memory_activation_list:List[List[str]]=field(default_factory=list)
    memory_param:List[str]=field(default_factory=list)
    memory_activation:List[str]=field(default_factory=list)
    memory_model=any
    memory_model_analysis=any
    #memory_model = memory_optimizer * memory_param + memory_activation
    pp_memory_model:List[str]=field(default_factory=list)
    comm_data_processor:CommDataProcessor=field(default_factory=CommDataProcessor)
    flow_graph:List[dict]=field(default_factory=list)
    mixed_precision: str = "default"  # 模型是否采用混精：default(weight_FB、weight、gradients)。默认采用。训练时权重和梯度为bfloat16。其中权重存在32位副本，梯度直接由32位采用优化强转，并在合并时强转回去
    search_fused_calculator_flag:bool =field(default_factory=bool)
    search_parallel_flag:bool=field(default_factory=bool)
    cost_model:List[dict]=field(default_factory=list)
    solution:List[int]=field(default_factory=list)
    profile_calc_Flag:bool=field(default_factory=bool)
    profile_comm_Flag:bool=field(default_factory=bool)
    flops:float=field(default_factory=float)
    module_auto:List[ModuleConfig]=field(default_factory=list)
    constant_auto:int=field(default_factory=int)
    acti_auto:List[str]=field(default_factory=list)
    param_auto:List[str]=field(default_factory=list)
    time_auto:List[float]=field(default_factory=list)
    acti_auto_address:List[str]=field(default_factory=list)
    front_acti_auto:List[str]=field(default_factory=list)
    front_param_auto:List[str]=field(default_factory=list)
    front_time_auto:List[float]=field(default_factory=list)
    middle_acti_auto:List[str]=field(default_factory=list)
    middle_param_auto:List[str]=field(default_factory=list)
    middle_time_auto:List[float]=field(default_factory=list)
    back_acti_auto:List[str]=field(default_factory=list)
    back_param_auto:List[str]=field(default_factory=list)
    back_time_auto:List[float]=field(default_factory=list)
    inter_time_auto:List[float]=field(default_factory=list)
    pipeline_time_auto:List[float]=field(default_factory=list)
    # 子类优化器
    optimizer=Optimizer()
    search_flag:bool=field(default_factory=bool)
    # -------------------- 计算方法 --------------------
    def __init__(self,args,mmlogs_path='',search_level=1,print_flag=False):
        self.args=args
        self.mla=False
        self.print_flag=print_flag
        self.calc_count=0
        self.profile_calc_Flag=False
        self.profile_comm_Flag=False
        self.included_modules=Included_Modules()
        self.included_modules_list=[]
        self.included_parallels_list=[]
        self.included_parallels=Included_Parallels()
        self.memory_param_list=[[],[],[]]
        self.moe_route_param_list=[]
        self.memory_activation_list=[[],[],[]]
        self.memory_param=[]
        self.moe_route_param=[]
        self.memory_activation=[]
        self.pp_memory_model=[]
        self.flow_graph=[]
        self.cost_model=[]
        self.solution=[]
        self.module_auto=[]
        self.constant_auto=0
        self.acti_auto=[]
        self.param_auto=[]
        self.time_auto=[]
        self.acti_auto_address=[]
        self.front_acti_auto=[]
        self.front_param_auto=[]
        self.front_time_auto=[]
        self.middle_acti_auto=[]
        self.middle_param_auto=[]
        self.middle_time_auto=[]
        self.back_acti_auto=[]
        self.back_param_auto=[]
        self.back_time_auto=[]
        self.inter_time_auto=[]
        self.pipeline_time_auto=[]
        self.num_layers_per_virtual_pipeline_stage=None
        #self.vpp=virtual_pipeline_model_parallel_size
        self.flag_moe=False
        self.shhffn=['seq','hidden','h_ffn']
        self.vocab=args.padded_vocab_size
        self.world_size = args.world_size#总卡数
        self.device_count =getattr(args, 'nproc_per_node', torch.cuda.device_count())
        #self.device_count = torch.cuda.device_count()#单节点上的卡数
        #self.device_count = args.nproc_per_node
        self.b=args.micro_batch_size
        self.seq=args.seq_length
        self.hidden=args.hidden_size
        self.head=args.num_attention_heads
        self.gbs=args.global_batch_size
        self.q_lora_rank=512 if getattr(args, 'q_lora_rank', None) is None else getattr(args, 'q_lora_rank', None)
        self.kv_lora_rank=512 if getattr(args, 'kv_lora_rank', None) is None else getattr(args, 'kv_lora_rank', None)
        self.qk_nope_head_dim=128 if getattr(args, 'qk_nope_head_dim', None) is None else getattr(args, 'qk_nope_head_dim', None)
        self.qk_rope_head_dim=64 if getattr(args, 'qk_rope_head_dim', None) is None else getattr(args, 'qk_rope_head_dim', None)
        self.v_head_dim=128 if getattr(args, 'v_head_dim', None) is None else getattr(args, 'v_head_dim', None)
        self.qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim
        
        if args.group_query_attention:
            self.group=args.num_query_groups
        else:
            self.group=self.head
        self.h_ffn=args.ffn_hidden_size
        self.moe_token_dispatcher_type=args.moe_token_dispatcher_type
        self.g=self.hidden/self.head
        if not self.h_ffn:
            self.h_ffn = 4 * self.hidden
        self.experts= 1 if args.num_experts is None else args.num_experts
        self.h_moe=getattr(args, 'moe_ffn_hidden_size', None)
        if self.h_moe is None:
            self.h_moe=self.h_ffn
        self.h_share_moe=getattr(args, 'moe_shared_expert_intermediate_size', None)
        if self.h_share_moe is None:
            self.share_experts=getattr(args, 'n_shared_experts', None)
            if self.share_experts is None: 
                self.share_experts = 1
            self.h_share_moe=self.share_experts*self.h_moe
        self.Topk=args.moe_router_topk
        # moe_expert_capacity_factor=1.0,感觉这个和下面重复了，废弃存疑
        moe_capacity=1.4 if getattr(args, 'moe_expert_capacity_factor', None) is None else args.moe_expert_capacity_factor#取默认值1.4
        self.capacity=moe_capacity
        self.layer=args.num_layers
        self.mmlogs_path=mmlogs_path
        self.search_level=search_level 
        if search_level==2 or search_level==4:
            self.map_manager = FileJSONHandler(f'{mmlogs_path}/calc_data/data.json',print_flag=print_flag)
            self.profile_calc_Flag=True
        if mmlogs_path !='' and (search_level==3 or search_level==4):
            self.comm_data_processor=CommDataProcessor(f'{mmlogs_path}/comm_data',print_flag=print_flag)
            self.profile_comm_Flag=True
        self.search_fused_calculator_flag = False
        self.search_module_flag = False
        self.search_parallel_flag = False
        #npu
        # if torch.npu.is_available():
        #     self.flops=NPU_TFLOPS
        #     cmd = "npu-smi info -t topo -i 0 | grep 'NPU0'"
        #     cmd2 = "npu-smi info -m | awk 'NR==2 {print $4,$5}'"
        #gpu
        if torch.cuda.is_available():#还没有验证过
            self.flops=GPU_TFLOPS
            # cmd = "gpu-smi info -t topo -i 0 | grep 'NPU0'"
            cmd2 = "nvidia-smi -i 0 --query-gpu=name --format=csv,noheader"
        else:#CPU
            self.flops=128*1e12#kupeng920
            # cmd = "echo cpu-single"
            cmd2 = "echo cpu"
        # result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        # self.device_topo=result.stdout.strip()
        result = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
        self.device_name=result.stdout.strip()
        self.search_model=None
        self.search_flag=False
       
    @classmethod
    def create(cls, *args, **kwargs):
        return cls(*args, **kwargs)
    @staticmethod
    def auto_analysis(mm_logs_path,args,parallel:str):
        acti_data = [0 for _ in range(args.world_size)]
        arch_data = [0 for _ in range(args.world_size)]
        time_data = [0 for _ in range(args.world_size)]
        # 使用os.walk递归遍历目录
        for root, dirs, files in os.walk(mm_logs_path):
            for file in files:
                if file.lower().endswith('.json'):  # 不区分大小写匹配
                    if 'arch_' not in file and 'acti_' not in file and 'time_' not in file: continue
                    file_path = os.path.join(root, file)
                    node_pattern = re.compile(r'node(\d+)')
                    node_number = int(node_pattern.search(os.path.splitext(file)[0]).group(1))
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'acti' in file_path:
                                acti_data[node_number]={
                                'path': file_path,
                                'content': data
                            }
                            elif 'arch' in file_path:
                                arch_data[node_number]={
                                'path': file_path,
                                'content': data
                            }
                            elif 'time' in file_path:
                                time_data[node_number]={
                                'path': file_path,
                                'content': data
                            }
                    except Exception as e:
                        print(f"读取文件 {file_path} 时出错: {str(e)}")
        #create_arch
        nm=NodeMerger()
        #获取本次记录时，所采用的，属于寻优空间的参数（如果寻优空间只采用几种并行，就只读取并行。如果寻优空间还打算加入vpp，那么本次运行的vpp也要记录,以便将采样值通过一系列计算，变成可以加任何影响后得出正确结论的基值）
        nm.merge_acti(acti_data)
        nm.merge_time(time_data)
        nm.merge_arch(arch_data,args=args)
        nm.model_init(parallel=parallel,args=args) 
        best_result=nm.auto_parallel()
        return best_result
    
    @staticmethod
    def set_opticonfig(args_fs,config:dict,distributed,hybrid,recompute,vpp):
        #bool
        args_fs.use_distributed_optimizer=config[distributed.instance_name]
        #int
        if config[vpp.instance_name] is not None:
            args_fs.num_layers_per_virtual_pipeline_stage=config[vpp.instance_name]
        else:
            args_fs.num_layers_per_virtual_pipeline_stage=None
        mla_flag = getattr(args_fs, 'multi_latent_attention', None)
        if mla_flag is None: mla_flag = getattr(args_fs, 'multi_head_latent_attention', False)
        if not mla_flag:
            if config[hybrid.instance_name] is not None and  config[hybrid.instance_name] != args_fs.num_attention_heads:
                args_fs.group_query_attention=True
                args_fs.num_query_groups=config[hybrid.instance_name]
            else:
                # Canonical MHA/off state: do not leave a stale GQA group count
                # from a previous strategy combination. This keeps strategy deltas
                # and experiment logs semantically clean.
                args_fs.group_query_attention=False
                args_fs.num_query_groups=args_fs.num_attention_heads
        #combine
        if config[recompute.instance_name] is not None:
            args_fs.recompute_granularity=recompute.recompute_granularity
            args_fs.recompute_modules=config[recompute.instance_name]
        else:
            # Canonical recompute-off state. The original implementation only
            # cleared granularity and could leave stale recompute_modules behind.
            args_fs.recompute_granularity=None
            args_fs.recompute_modules=None
        return args_fs
    
    @classmethod
    def search_addoptispace_create(cls,args,mmlogs_path,search_level=1,print_flag=False,single_node=True,cpu_only=False):
        all_rank=min(args.world_size,torch.cuda.device_count())
        if args.rank / all_rank >= 1: return None,None,None,None,None#有一个节点运行就可以了
        rank = args.rank % all_rank
        if cpu_only:
            all_rank=1
            rank=0
        resultmap=FileJSONHandler(f'{mmlogs_path}/search_data/result_node{rank}.json',init_flag=False)
        sonmap:list=[]
        args_fs=copy.deepcopy(args)

        vpp=VirtualPipe(num_layers=args.num_layers)
        recompute=ReCompute()
        distributed=DistributedOptimizer()
        hybrid=Hybrid_MHA_MQA()
        namelist=[]
        optilist=[]
    
        namelist.append(recompute.instance_name)
        configvalue=recompute.set_valuerange()
        #识别架构，特殊处理一下
        if args.num_experts is None:
            configvalue[1]=[d for d in configvalue[1] if d != "moe"]
        optilist.append(configvalue)

        namelist.append(vpp.instance_name)
        optilist.append(vpp.set_valuerange())
        namelist.append(distributed.instance_name)
        optilist.append(distributed.set_valuerange())
        mla_flag = getattr(args, 'multi_latent_attention', None)
        if mla_flag is None: mla_flag = getattr(args, 'multi_head_latent_attention', False)
        if not mla_flag:
            namelist.append(hybrid.instance_name)
            hybrid_value=hybrid.set_valuerange(args.num_attention_heads)
            if args_fs.group_query_attention and args_fs.num_query_groups < hybrid_value[1][0]:
                hybrid_value[1][0]=args_fs.num_query_groups
            optilist.append(copy.deepcopy(hybrid_value))
        if rank == 0 :
            for name in namelist:
                print(f'search optimization :{name}')
        for idx in optilist:
            if idx[0] == 'bool':
                continue#直接写的时候就弄好了
            elif idx[0] == 'int':
                divisor=1
                temp:List[int]=[]
                while divisor <= idx[1][1]:#由上限慢慢加到下限
                    value=int(idx[1][1]//divisor)
                    if idx[1][1]%divisor == 0 and value >= idx[1][0]:
                        temp.append(value)
                    divisor+=1
                idx[1]=temp
            elif idx[0] == 'combine':
                temp=[]
                for r in range(1, len(idx[1]) + 1):  # 遍历所有可能的组合长度
                    combos = itertools.combinations(idx[1], r)
                    temp.extend([[*combo] for combo in combos])  # 将元组转换为列表
                temp.append(None)
                idx[1]= temp
            # elif idx[0] == 'int':
            #     temp:List[int]=[]
            #     while idx[1][1] > idx[1][0]:
            #         temp.append(idx[1][1])
            #         idx[1][1]=idx[1][1]//2
            #     temp.append(idx[1][0])
            #     temp.append(None)
            #     idx[1]=temp
        # if not args.multi_latent_attention:
        #     index=namelist.index(hybrid.instance_name)
        #     hybrid_value[1][0]=optilist[index][1][-1]
        # 提取键名和对应的列表
        keys = [pair for pair in namelist]
        value_lists = [pair[1] for pair in optilist]
        # VPP is an optional strategy. The original code enumerated only enabled
        # VPP layer sizes; TPDS experiments also need an explicit disabled state
        # so semantic-equivalence and compound-effect tests have a real baseline.
        if vpp.instance_name in keys:
            vpp_idx = keys.index(vpp.instance_name)
            if None not in value_lists[vpp_idx]:
                value_lists[vpp_idx] = list(value_lists[vpp_idx]) + [None]
        # 生成笛卡尔积
        combinations = itertools.product(*value_lists)
        # 构建动态字典列表
        search_args = [dict(zip(keys, combo)) for combo in combinations]
        # Single-GPU correctness mode: VPP requires PP>1 and distributed optimizer
        # has no sharding benefit at DP=1. Exclude these no-op/invalid dimensions so
        # the small smoke set focuses on meaningful GQA and recomputation deltas.
        if TPDS_RUNTIME.config.active and TPDS_RUNTIME.config.single_gpu_smoke:
            search_args = [c for c in search_args
                           if c.get("VirtualPipe") is None
                           and not bool(c.get("DistributedOptimizer", False))]
        if TPDS_RUNTIME.config.active and TPDS_RUNTIME.config.strategy_limit > 0:
            search_args = _tpds_select_strategy_combos(
                search_args, TPDS_RUNTIME.config.strategy_limit,
                seed=TPDS_RUNTIME.config.sample_seed,
                mode=TPDS_RUNTIME.config.strategy_selection)
        if rank == 0 :
            print(f'search optimization combine num :{len(search_args)}')
            print(f"max_available_memory: {torch.cuda.get_device_properties(0).total_memory}")
        #import pdb; pdb.set_trace()
        for i in range(len(search_args)):
            if (i % all_rank) == rank:
                args_fs=GPT.set_opticonfig(args_fs,search_args[i],distributed,hybrid,recompute,vpp)
                if TPDS_RUNTIME.config.active:
                    TPDS_RUNTIME.note_strategy(args_fs)
                start_time = time.time()
                search_model=cls.create(args_fs,mmlogs_path,search_level)
                solutions=search_model.search_space_create()
                if len(solutions) == 0: continue
                time_best,s_best=search_model.costmodel_create(solutions)
                end_time=time.time() - start_time
                print(f"rank{rank}:find search optimization combine-{i}:{search_args[i]},optimal configuration: {s_best}, find optiaml cost:{time_best},search_cost_time: {end_time} \n")             
                if args_fs.num_layers_per_virtual_pipeline_stage is not None:
                    # if args_fs.num_layers%(args_fs.num_layers_per_virtual_pipeline_stage * s_best[1]) != 0: 
                    #     vppnum=args_fs.num_layers//(args_fs.num_layers_per_virtual_pipeline_stage * s_best[1])
                    #     if vppnum >1:
                    #         search_args[i][vpp.instance_name] = args_fs.num_layers//(vppnum * s_best[1])
                    #     else:
                    #         search_args[i][vpp.instance_name] = None
                    
                    if args_fs.num_layers%(args_fs.num_layers_per_virtual_pipeline_stage * s_best[1]) != 0 or s_best[1] == 1 or args_fs.num_layers <= args_fs.num_layers_per_virtual_pipeline_stage * s_best[1]:#vpp不符合要求时，实际跑会出问题
                        search_args[i][vpp.instance_name] = None
                
                sonmap.append({'combine':search_args[i],'configuration':s_best,'cost':time_best,'search_cost_time':end_time,'solutionsnum':len(solutions)})
        resultmap.data_map['result']=sonmap
        son_len=len(sonmap)
        best_combine=[sonmap[-1]['combine']] if son_len >0 else {}
        best_conf=[sonmap[-1]['configuration']] if son_len >0 else []
        best_cost_time=sonmap[-1]['cost'] if son_len >0 else sys.float_info.max
        best_search_time=sonmap[-1]['search_cost_time'] if son_len >0 else sys.float_info.max
        best_num_solutions=sonmap[-1]['solutionsnum'] if son_len >0 else 0
        resultmap._save_to_json()

        # 构建匹配模式：result_node后跟一个或多个数字，然后是.json
        pattern = os.path.join(f'{mmlogs_path}/search_data', "result_node[0-9]*.json")
        
        # 查找所有匹配的文件
        matching_files = glob.glob(pattern)
        
        # 进一步精确过滤，确保数字后面直接跟着.json（避免匹配如 result_node123abc.json 这样的文件）
        precise_matches = [
            f for f in matching_files 
            if os.path.basename(f).startswith("result_node") and 
            os.path.basename(f)[11:-5].isdigit() and  # 提取"result_node"和".json"之间的部分检查是否为纯数字
            f.endswith(".json")
        ]
        file_count = len(precise_matches)
        if file_count < all_rank : return None,None,None,None,None
        #只有最后的rank进入以下循环
        all_search_time=0.0
        all_solutionsnum=0
        for root, dirs, files in os.walk(f'{mmlogs_path}/search_data'):
            for file in files:
                file_path = os.path.join(root, file)
                # node_pattern = re.compile(r'node(\d+)')
                # node_number = int(node_pattern.search(os.path.splitext(file)[0]).group(1))
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for idx in data['result']:
                            all_search_time=all_search_time+idx['search_cost_time']
                            all_solutionsnum=all_solutionsnum+idx['solutionsnum']
                            if idx['cost'] < best_cost_time or (idx['cost'] == best_cost_time and idx['search_cost_time'] < best_search_time):
                                best_combine=[idx['combine']]
                                best_conf=[idx['configuration']]
                                best_cost_time=idx['cost']
                                best_search_time=idx['search_cost_time']
                                best_num_solutions=idx['solutionsnum']
                            elif idx['cost'] == best_cost_time and idx['combine'] not in best_combine :
                                best_combine.append(idx['combine'])
                                best_conf.append(idx['configuration'])
                except Exception as e:
                    print(f"rank{rank}读取文件 {file_path} 时出错: {str(e)}")

        formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 输出: 2025年09月28日
        resultmap.data_map={}
        resultmap.file_path=f'{mmlogs_path}/optisearch_result_{formatted}.json'
        
        resultmap.data_map['combine']=best_combine
        resultmap.data_map['configuration']=best_conf
        resultmap.data_map['solutionsnum']=best_num_solutions
        resultmap.data_map['traincost']=best_cost_time
        resultmap.data_map['searchcost']=best_search_time

        resultmap.data_map['combine_num']=len(search_args)
        resultmap.data_map['all_solutions_num']=all_solutionsnum
        resultmap.data_map['all_searchcost']=all_search_time
        resultmap.data_map['avg_searchcost']=all_search_time/len(search_args)

        resultmap._save_to_json()
        # if self.args_fs.rank !=0: time.sleep(360000)
        # import pdb; pdb.set_trace()  
        # if not args.multi_latent_attention and best_combine[0][hybrid.instance_name] is None or best_combine[0][hybrid.instance_name] > hybrid_value[1][0] and hybrid_value[1][0] % best_conf[0][4] == 0:
        #     best_combine[0][hybrid.instance_name] = hybrid_value[1][0]
        # if best_conf[0][0] > 1 :
        #     best_combine[0][distributed.instance_name] =True
        if len(best_conf) > 0:
            print(f"find num:{len(search_args)},the lastoptimal combine:{best_combine[0]},configuration: {best_conf[0]}, find optiaml cost:{best_cost_time},search_cost_time: {best_search_time}")
            best_args=GPT.set_opticonfig(args_fs,best_combine[0],distributed,hybrid,recompute,vpp)
            best_args.data_parallel_size=best_conf[0][0]
            best_args.pipeline_model_parallel_size=best_conf[0][1]
            best_args.context_parallel_size=best_conf[0][2]*best_conf[0][3]
            best_args.tensor_model_parallel_size=best_conf[0][4]
            best_args.expert_model_parallel_size=best_conf[0][6]
            best_args.micro_batch_size=best_conf[0][7]
            return best_args,best_conf[0],best_cost_time,best_search_time,rank
        else:
            return None,None,None,None,rank
        
    def get_module_auto(self,name:str):
        for idx in self.module_auto:
            if idx.instance_name == name:
                return idx
        return None
    
    def init_parallel(self):
        args=self.args
          # 设置并行参数
        #listop.append(OptimizerCollection.parallel.datapara.distributedoptimizer())
        listop=[]
        if self.args.use_distributed_optimizer:
            listop.append(OptimizerCollection.parallel.datapara.distributedoptimizer())
        self.included_parallels.DataParallel(args,optimizations=copy.deepcopy(listop))
        listop.clear()
        
        #目前还不清楚虚拟流水线的代价是什么，只考虑了pp变多后，stage间recv+send的影响，导致结果不准，vpp越高，time约等于time/vpp
        self.num_layers_per_virtual_pipeline_stage = getattr(args, 'num_layers_per_virtual_pipeline_stage', None)
        if self.num_layers_per_virtual_pipeline_stage is not None:
           listop.append(OptimizerCollection.parallel.pipepara.virtualpipe(args.num_layers,self.num_layers_per_virtual_pipeline_stage))
        self.included_parallels.PipeParallel(args,optimizations=copy.deepcopy(listop))
        listop.clear()
        
        self.included_parallels.ContextParallel(args)
        self.included_parallels.UlyssessParallel(args)
        #默认开启sp
        self.included_parallels.TensorParallel(args)
        self.included_parallels.ExpertParallel(args)
        #import pdb; pdb.set_trace() 
    
    
    def _init_modules(self, args):    
        return 0
    
    def search_fused_calculator(self):
        return 0
    
    def search_module(self):
        return 0
    
    def search_parallel(self):
        return 0
    def calculate_model(self) -> str:
        """
        {梯度+权重+优化器}{模型参数[前向模块+中间模块*L+后向模块]+激活值[前向模块+中间模块*L+后向模块]}
        在融合算子替代之后
        先确定每个模块的最终表现形式，然后按照DP-PP-CP&UP-TP&SP的形式，在相应的位置进行划分，划分的依据则是：XX并行.影响的模块
        """
        i:int =0
        idx:int = 0
        layer_graph=[] 
        for module in self.included_modules.front:
            layer_graph.append({"name":f"front-{module.instance_name}-index-input",
                                        "parallel":"",
                                        "num":i,
                                        "values":module.input_shape,
                                        "parallel_mode":"",
                                        "left-matrix":module.input_shape,
                                        "right-matrix":"",
                                        "calc":"0",
                                        "primitive":[],
                                        "com_data":[],
                                        "idx":idx,
                                        'param':'0',
                                        'to_memory':0,
                                        "recompute":module.recompute_flag})
            idx=idx+1
            staitc_param,moe_param=module.get_module_params()
            self.memory_param_list[0].append(staitc_param)
            self.memory_activation_list[0].append(module.get_activation())
            for impact in module.parallel_impact:
                if "tp" in impact["parallel"] and "tpf" not in impact["parallel"]: 
                    mode=impact["parallel_mode"]
                    w_param=impact['param']
                else: 
                    mode=""
                    w_param="0"
                layer_graph.append({"name":f"front-{module.instance_name}-{impact['type']}",
                                            "parallel":impact["parallel"],
                                            "num":i,
                                            "values":impact["values"],
                                            'param':w_param,
                                            "parallel_mode":mode,
                                            "left-matrix":impact["left-matrix"],
                                            "right-matrix":impact["right-matrix"],
                                            "calc":impact["calc"],
                                            "primitive":[],
                                            "com_data":[],
                                            "idx":idx,
                                            'to_memory':impact['to_memory'],
                                            "recompute":module.recompute_flag})
                idx=idx+1
            layer_graph.append({"name":f"front-{module.instance_name}-index-output",
                                        "parallel":"",
                                        "num":i,
                                        "values":module.output_shape,
                                        "parallel_mode":"",
                                        "left-matrix":module.output_shape,
                                        "right-matrix":"",
                                        "calc":"0",
                                        "primitive":[],
                                        "com_data":[],
                                        "idx":idx,
                                        'param':'0',
                                        'to_memory':0,
                                        "recompute":module.recompute_flag})
            idx=idx+1
            i=i+1
        self.flow_graph.append(layer_graph)
        self.memory_param.append("+".join(self.memory_param_list[0])) 
        self.memory_activation.append("+".join(self.memory_activation_list[0]))
        for layer in self.included_modules.middle:
            layer_graph=[]
            layer_param=[]
            moe_route_layer_param=[]
            layer_acti=[]
            for module in layer:
                layer_graph.append({"name":f"middle-{module.instance_name}-index-input",
                                            "parallel":"",
                                            "num":i,
                                            "values":module.input_shape,
                                            "parallel_mode":"",
                                            "left-matrix":module.input_shape,
                                            "right-matrix":"",
                                            "calc":"0",
                                            "primitive":[],
                                            "com_data":[],
                                            "idx":idx,
                                            'param':'0',
                                            'to_memory':0,
                                            "recompute":module.recompute_flag})
                idx=idx+1
                staitc_param,moe_param=module.get_module_params()
                layer_param.append(staitc_param)
                if moe_param != '()':
                    moe_route_layer_param.append(moe_param)
                else:
                    moe_route_layer_param.append('0')
                layer_acti.append(module.get_activation())
                for impact in module.parallel_impact:
                    if "tp" in impact["parallel"] and "tpf" not in impact["parallel"]: 
                        mode=impact["parallel_mode"]
                        w_param=impact['param']
                    else: 
                        mode=""
                        w_param="0"
                    layer_graph.append({"name":f"middle-{module.instance_name}-{impact['type']}",
                                                "parallel":impact["parallel"],
                                                "num":i,
                                                "values":impact["values"],
                                                "parallel_mode":mode,
                                                'param':w_param,
                                                "left-matrix":impact["left-matrix"],
                                                "right-matrix":impact["right-matrix"],
                                                "calc":impact["calc"],
                                                "primitive":[],
                                                "com_data":[],
                                                "idx":idx,
                                                'to_memory':impact['to_memory'],
                                                "recompute":module.recompute_flag})
                    idx=idx+1
                layer_graph.append({"name":f"middle-{module.instance_name}-index-output",
                                            "parallel":"",
                                            "num":i,
                                            "values":module.output_shape,
                                            "parallel_mode":"",
                                            "left-matrix":module.output_shape,
                                            "right-matrix":"",
                                            "calc":"0",
                                            "primitive":[],
                                            "com_data":[],
                                            "idx":idx,
                                            'param':'0',
                                            'to_memory':0,
                                            "recompute":module.recompute_flag})
                idx=idx+1
                i=i+1 
            self.flow_graph.append(layer_graph)
            self.memory_param_list[1].append(f"({'+'.join(layer_param)})")
            self.moe_route_param_list.append(f"({'+'.join(moe_route_layer_param)})")
            self.memory_activation_list[1].append(f"({'+'.join(layer_acti)})")
        self.memory_param.append(f"{'&'.join(self.memory_param_list[1])}")
        self.moe_route_param.append(f"{'&'.join(self.moe_route_param_list)}")
        self.memory_activation.append(f"{'&'.join(self.memory_activation_list[1])}") 
        layer_graph=[]
        for module in self.included_modules.back:
            layer_graph.append({"name":f"back-{module.instance_name}-index-input",
                                        "parallel":"",
                                        "num":i,
                                        "values":module.input_shape,
                                        "parallel_mode":"",
                                        "left-matrix":module.input_shape,
                                        "right-matrix":"",
                                        "calc":"0",
                                        "primitive":[],
                                        "com_data":[],
                                        "idx":idx,
                                        'param':'0',
                                        'to_memory':0,
                                        "recompute":module.recompute_flag})
            idx=idx+1
            staitc_param,moe_param=module.get_module_params()
            self.memory_param_list[2].append(staitc_param)
            self.memory_activation_list[2].append(module.get_activation())
            for impact in module.parallel_impact:
                if "tp" in impact["parallel"] and "tpf" not in impact["parallel"]: 
                    mode=impact["parallel_mode"]
                    w_param=impact['param']
                else: 
                    mode=""
                    w_param="0"
                layer_graph.append({"name":f"back-{module.instance_name}-{impact['type']}",
                                            "parallel":impact["parallel"],
                                            "num":i,
                                            "values":impact["values"],
                                            "parallel_mode":mode,
                                            'param':w_param,
                                            "left-matrix":impact["left-matrix"],
                                            "right-matrix":impact["right-matrix"],
                                            "calc":impact["calc"],
                                            "primitive":[],
                                            "com_data":[],
                                            "idx":idx,
                                            'to_memory':impact['to_memory'],
                                            "recompute":module.recompute_flag})
                idx=idx+1
            layer_graph.append({"name":f"back-{module.instance_name}-index-output",
                                        "parallel":"",
                                        "num":i,
                                        "values":module.output_shape,
                                        "parallel_mode":"",
                                        "left-matrix":module.output_shape,
                                        "right-matrix":"",
                                        "calc":"0",
                                        "primitive":[],
                                        "com_data":[],
                                        "idx":idx,
                                        'param':'0',
                                        'to_memory':0,
                                        "recompute":module.recompute_flag})
            idx=idx+1
            i=i+1
        self.flow_graph.append(layer_graph)
        self.memory_param.append("+".join(self.memory_param_list[2])) 
        self.memory_activation.append("+".join(self.memory_activation_list[2])) 
    
    def memory_model_create(self):
        if self.search_fused_calculator_flag == True:
            print("") ##生成包含有融合算子的included_modules，并加入included_modules_list
        if self.search_module_flag == True:
            print("")##遍历included_modules_list，基于每一个included_modules，生成有不同优化选项的included_modules，并加入included_modules_list
        if self.search_parallel_flag == True:
            print("")##生成具有不同并行优化的included_parallel,并加入included_parallel_list
        #计算内存模型
        self.module_rename()
        self.calculate_model()

    def module_rename(self):
        i = 0
        remap={}
        for item in self.included_modules.front+self.included_modules.middle+self.included_modules.back:
            if type(item) == list:
                for idx in item:
                    if idx.instance_name in remap:
                        idx.rename(f"{idx.instance_name}{i}")
                        i=i+1
                    else:
                        remap[idx.instance_name]=True
            else: 
                if item.instance_name in remap:
                    item.rename(f"{item.instance_name}{i}")
                    i=i+1
                else:
                    remap[item.instance_name]=True
    
    def auto_memory_model_create_pp(self,pp):
        pp_model=[]
        pp_model_analysis=[]
        if pp > 1:
            param_auto_list=[sub.tolist() for sub in np.array_split(self.middle_param_auto, pp)]
            acti_auto_list=[sub.tolist() for sub in np.array_split(self.middle_acti_auto, pp)]
            param_auto_list[0]=self.front_param_auto+param_auto_list[0]
            acti_auto_list[0]=self.front_acti_auto+acti_auto_list[0]
            param_auto_list[-1]=param_auto_list[-1]+self.back_param_auto
            acti_auto_list[-1]=acti_auto_list[-1]+self.back_acti_auto
            numlist=self.included_parallels.pipe_p.calc_actinum(pp)
            for i in range(len(numlist)):
                num=numlist[i]
                pp_model.append(f"{self.memory_optimizer}*({'+'.join(param_auto_list[i])})\
                                   +({num}*({'+'.join(acti_auto_list[i])}))")
                pp_model_analysis.append([f"{self.memory_optimizer}*({'+'.join(param_auto_list[i])})",f"{num}*({'+'.join(acti_auto_list[i])})"])
            self.memory_model=pp_model
            self.memory_model_analysis=pp_model_analysis
        else:
            self.memory_model=f"{self.memory_optimizer}*\
                ({'+'.join(self.front_param_auto+self.middle_param_auto+self.back_param_auto)})+\
                ({'+'.join(self.front_acti_auto+self.middle_acti_auto+self.back_acti_auto)})"
            self.memory_model_analysis=[f"{self.memory_optimizer}*\
                ({'+'.join(self.front_param_auto+self.middle_param_auto+self.back_param_auto)})",f"{'+'.join(self.front_acti_auto+self.middle_acti_auto+self.back_acti_auto)}"]
        return '+'.join(self.middle_param_auto)
    def memory_model_create_pp(self,pp):
        pp_model=[]
        pp_analysis=[]
        memory_param=copy.deepcopy(self.memory_param)
        memory_activation=copy.deepcopy(self.memory_activation)
        moe_route_param=copy.deepcopy(self.moe_route_param)
        param_str='0'
        pp_param,pp_acti,pp_moe_route_param =self.included_parallels.pipe_p.memory_apply(pp,memory_param,memory_activation,moe_route_param)
        moe_result=self.eval_calc('+'.join(pp_moe_route_param[0]),[1,1,1,1,1,1,1,1,1])
        for i in range(pp_param.__len__()):
            if moe_result == 0:
                pp_model.append(f"{self.memory_optimizer}*({'+'.join(pp_param[i])})+({'+'.join(pp_acti[i])})")
                pp_analysis.append([f"{self.memory_optimizer}*({'+'.join(pp_param[i])})",f"{'+'.join(pp_acti[i])}"])
                param_str=f"{param_str}+({'+'.join(pp_param[i])})"
            else:
                pp_model.append(f"{self.moe_route_optimizer}*({'+'.join(pp_moe_route_param[i])})+{self.memory_optimizer}*({'+'.join(pp_param[i])})+({'+'.join(pp_acti[i])})")
                pp_analysis.append([f"{self.moe_route_optimizer}*({'+'.join(pp_moe_route_param[i])})+{self.memory_optimizer}*({'+'.join(pp_param[i])})",f"{'+'.join(pp_acti[i])}"])
                param_str=f"{param_str}+({'+'.join(pp_moe_route_param[i])}+{'+'.join(pp_param[i])})"
        self.memory_model=pp_model
        self.memory_model_analysis=pp_analysis
        return param_str
    
    def _tpds_structural_key(self):
        """Facts that affect structural candidate enumeration, but not memory cost."""
        return (
            self.world_size, self.device_count, self.layer, self.gbs, self.seq,
            self.head, self.group, self.experts, bool(self.mla),
        )

    def _tpds_memory_signature(self):
        recompute_modules = getattr(self.args, 'recompute_modules', None)
        if isinstance(recompute_modules, list):
            recompute_modules = tuple(recompute_modules)
        return (
            getattr(self.args, 'recompute_granularity', None),
            recompute_modules,
            getattr(self.args, 'recompute_method', None),
            getattr(self.args, 'recompute_num_layers', None),
            bool(getattr(self.args, 'use_distributed_optimizer', False)),
            getattr(self.args, 'num_layers_per_virtual_pipeline_stage', None),
            bool(getattr(self.args, 'group_query_attention', False)),
            getattr(self.args, 'num_query_groups', None),
        )

    def _tpds_cost_signature(self):
        """Dependencies of schedule/operator cost in the current implementation.

        DistributedOptimizer is intentionally excluded because the current DP flow
        model assigns it no time penalty; it only changes persistent memory.
        """
        recompute_modules = getattr(self.args, 'recompute_modules', None)
        if isinstance(recompute_modules, list):
            recompute_modules = tuple(recompute_modules)
        return (
            self.layer, self.seq, self.hidden, self.head, self.group, self.h_ffn,
            self.experts, bool(self.mla), bool(getattr(self.args, 'use_flash_attn', False)),
            getattr(self.args, 'recompute_granularity', None), recompute_modules,
            getattr(self.args, 'num_layers_per_virtual_pipeline_stage', None),
            bool(getattr(self.args, 'group_query_attention', False)),
            getattr(self.args, 'num_query_groups', None),
        )

    def _tpds_enumerate_structural_candidates(self):
        solutions = []
        for pp in range(1, self.world_size + 1):
            if self.world_size % pp != 0 or self.layer < pp or self.layer % pp != 0:
                continue
            for i in range(self.device_count):
                tp = 2 ** i
                if tp > self.device_count or tp > (self.world_size // pp):
                    break
                if (self.group > 1 and self.group % tp != 0) or (self.head % tp != 0):
                    break
                max_cp_size = self.world_size // (pp * tp)
                for cp_size in range(1, max_cp_size + 1):
                    if self.world_size % (pp * tp * cp_size) != 0 or self.gbs % (self.world_size // (pp * tp * cp_size)) != 0:
                        continue
                    # GPU path: ulysses is currently fixed to one, matching the original code.
                    for ulyp in range(1, 2):
                        if cp_size % ulyp != 0:
                            continue
                        cp = cp_size // ulyp
                        head, remainder = divmod(self.head, ulyp * tp)
                        if (head < 1 or remainder != 0) or (self.seq % (2 * cp) != 0) or (self.group > 1 and self.group % (ulyp * tp) != 0):
                            continue
                        dp = self.world_size // (pp * tp * cp_size)
                        dp_group_batch_size = self.gbs // dp
                        max_ep_size = min(self.world_size // (pp * tp), self.experts)
                        for ep in range(1, max_ep_size + 1):
                            if self.experts % ep != 0 or (dp * cp_size) % ep != 0:
                                continue
                            for num_mb in range(1, dp_group_batch_size + 1):
                                if num_mb < pp or dp_group_batch_size % num_mb != 0:
                                    continue
                                mbs = dp_group_batch_size // num_mb
                                if TPDS_RUNTIME.config.active and TPDS_RUNTIME.config.max_mbs > 0 and mbs > TPDS_RUNTIME.config.max_mbs:
                                    continue
                                sp = tp
                                if self.mla and (cp > 1 or tp > 1 or ulyp > 1):
                                    continue
                                solutions.append([dp, pp, cp, ulyp, tp, sp, ep, mbs, num_mb])
        return solutions

    def search_space_create(self,precent=0.9,auto_flag=False):
        # Preserve the original memory semantics while separating structural
        # enumeration from memory feasibility. This is the first real incremental
        # reuse boundary: strategy changes that do not affect structural facts can
        # reuse the structural candidate set.
        self.memory_optimizer,self.moe_route_optimizer=self.included_parallels.data_p.memory_apply(self.memory_optimizer,self.moe_route_optimizer)
        properties = torch.cuda.get_device_properties(0)
        max_available_memory = properties.total_memory
        if self.print_flag:
            print(self.memory_optimizer)
            print(f"max_available_memory: {max_available_memory},memory_use_precent:{precent}")

        runtime = TPDS_RUNTIME
        structural_key = self._tpds_structural_key()
        t0 = time.perf_counter()
        if runtime.config.active and runtime.config.use_incremental and structural_key in runtime.structural_space_cache:
            structural = copy.deepcopy(runtime.structural_space_cache[structural_key])
            runtime.analysis('structural_space', time.perf_counter()-t0, cache_hit=True)
        else:
            structural = self._tpds_enumerate_structural_candidates()
            if runtime.config.active and runtime.config.use_incremental:
                runtime.structural_space_cache[structural_key] = copy.deepcopy(structural)
            if runtime.config.active:
                runtime.analysis('structural_space', time.perf_counter()-t0, cache_hit=False)
        if runtime.config.active:
            # VPP is executable only with PP>1 and at least two equal virtual
            # chunks per physical pipeline rank. Filter illegal combinations
            # before memory/cost evaluation; the VPP-disabled strategy covers
            # the corresponding canonical configuration.
            vpp_setting = runtime.current_strategy.get('VirtualPipe')
            if vpp_setting not in (None, 0, 'None'):
                vpp_setting = int(vpp_setting)
                layers = int(getattr(self.args, 'num_layers', self.layer))
                before_legality = len(structural)
                structural = [s for s in structural
                              if int(s[1]) > 1
                              and layers % (int(s[1]) * vpp_setting) == 0
                              and layers > int(s[1]) * vpp_setting]
                runtime.stats['candidates_rejected_legality'] += before_legality - len(structural)
            runtime.stats['structural_candidates'] += len(structural)

        solutions = []
        memory_templates = {}
        memory_sig = self._tpds_memory_signature()
        for s in structural:
            pp = s[1]
            if runtime.config.active and runtime.config.use_incremental:
                cache_key = (self._tpds_structural_key(), self.hidden, self.h_ffn, self.vocab, memory_sig, pp, bool(auto_flag))
                if cache_key in runtime.memory_template_cache:
                    memory_model, memory_analysis = runtime.memory_template_cache[cache_key]
                    self.memory_model = copy.deepcopy(memory_model)
                    self.memory_model_analysis = copy.deepcopy(memory_analysis)
                    runtime.stats['memory_template_cache_hits'] += 1
                else:
                    mt0 = time.perf_counter()
                    if auto_flag:
                        self.auto_memory_model_create_pp(pp)
                    else:
                        self.memory_model_create_pp(pp)
                    runtime.analysis('memory_template', time.perf_counter()-mt0)
                    runtime.memory_template_cache[cache_key] = (copy.deepcopy(self.memory_model), copy.deepcopy(self.memory_model_analysis))
            else:
                mt0 = time.perf_counter()
                if auto_flag:
                    self.auto_memory_model_create_pp(pp)
                else:
                    self.memory_model_create_pp(pp)
                if runtime.config.active:
                    runtime.analysis('memory_template', time.perf_counter()-mt0)

            mt0 = time.perf_counter()
            peak = 0
            for part in self.memory_model:
                value = self.eval_calc(part, s)
                if peak == 0 or peak < value:
                    peak = value
            if runtime.config.active:
                runtime.analysis('memory_feasibility', time.perf_counter()-mt0)
            if peak < precent * max_available_memory:
                solutions.append(s)
            elif runtime.config.active:
                runtime.record_rejected({
                    'parallel': s,
                    'strategy': runtime.current_strategy,
                    'predicted_feasible': False,
                    'predicted_peak_memory': peak,
                    'memory_limit': precent * max_available_memory,
                    'reason': 'predicted_oom',
                })
        if runtime.config.active:
            runtime.stats['feasible_candidates'] += len(solutions)
        if self.print_flag:
            print(f"filter search_space: {len(solutions)}")
            for a in solutions:
                print(f"[{a[0]},{a[1]},{a[2]},{a[3]},{a[4]},{a[5]},{a[6]},{a[7]},{a[8]}]")
        return solutions

    def print_bestresult(self,s,auto_flag=False,print_flag=True):
        if print_flag:
            print(s)
        pp=s[1]
        if auto_flag:
            self.auto_memory_model_create_pp(pp)
        else:
            self.memory_model_create_pp(pp)
        result:float=0.0
        result_ptmp:float=0.0
        result_atmp:float=0.0
        #result=result_param=result_acti=result_ptmp=result_atmp=0
        for i in range(len(self.memory_model_analysis)):
            result_ptmp=self.eval_calc(self.memory_model_analysis[i][0],s,print_flag,name=f'param{i}') 
            result_atmp=self.eval_calc(self.memory_model_analysis[i][1],s,print_flag,name=f'acti{i}')     
            if result == 0 or result<(result_ptmp+result_atmp):
                result= result_ptmp+result_atmp
        if print_flag:
            print(f'max_result:{result}={result/1024/1024}MB={result/1024/1024/1024}GB')  
    def costmodel_create(self,solutions:List[List[int]],auto_flag=False):
        runtime = TPDS_RUNTIME
        time_best=0
        eval_solutions = list(solutions)
        limit = runtime.config.candidate_limit if runtime.config.active else 0
        if limit > 0 and len(eval_solutions) > limit:
            # Deterministic sampling is preferable to taking only the first N,
            # because the original enumeration order is structured by PP/TP/CP.
            rng = random.Random(runtime.config.sample_seed)
            indices = sorted(rng.sample(range(len(eval_solutions)), limit))
            eval_solutions = [eval_solutions[i] for i in indices]
        if runtime.config.active:
            runtime.stats['candidate_pool_before_limit'] += len(solutions)
            runtime.stats['candidate_pool_after_limit'] += len(eval_solutions)

        cost_sig = self._tpds_cost_signature()
        for s in eval_solutions:
            cache_key = (cost_sig, tuple(s), bool(auto_flag))
            if runtime.config.active and runtime.config.use_incremental and cache_key in runtime.cost_cache:
                timecost = runtime.cost_cache[cache_key]
                runtime.analysis('schedule_cost', cache_hit=True)
                cm_temp = None
            else:
                ct0 = time.perf_counter()
                if auto_flag:
                    if s[1] > 1:
                        cm_temp=[sub.tolist() for sub in np.array_split(self.middle_time_auto, s[1])]
                        cm_temp[0]=self.front_time_auto+cm_temp[0]
                        cm_temp[-1]=cm_temp[-1]+self.back_time_auto
                    else:
                        cm_temp=[self.front_time_auto+self.middle_time_auto+self.back_time_auto]
                else:
                    cm_temp=copy.deepcopy(self.flow_graph)
                    cm_for_deal = [item for sublist in cm_temp for item in sublist]
                    if s[6] > 1:
                        self.included_parallels.expert_p.EP_flowgraph(cm_for_deal, s[6])
                    if s[3] > 1:
                        self.included_parallels.ulyssess_p.UP_flowgraph(cm_for_deal,s[3])
                    if s[2] > 1:
                        self.included_parallels.context_p.CP_flowgraph(cm_for_deal, s[2])
                    if s[4] > 1:
                        self.included_parallels.tensor_p.TP_SP_flowgraph(cm_for_deal, s[4], s[5])
                    if s[1] > 1:
                        self.included_parallels.pipe_p.PP_flowgraph(cm_for_deal, s[1])
                    if s[0] > 1:
                        self.included_parallels.data_p.DP_flowgraph(cm_for_deal, s[0])
                timecost=self.costmodel_timecost(s,cm_temp,auto_flag)
                if runtime.config.active:
                    runtime.analysis('schedule_cost', time.perf_counter()-ct0)
                    if runtime.config.use_incremental:
                        runtime.cost_cache[cache_key]=timecost
            if self.print_flag:
                print(timecost)
            if time_best ==0 or time_best > timecost:
                time_best=timecost
                self.solution=s
                if cm_temp is not None:
                    self.cost_model=cm_temp
            if runtime.config.active:
                runtime.record_candidate({
                    'parallel': s,
                    'strategy': runtime.current_strategy,
                    'predicted_cost': timecost,
                    'predicted_feasible': True,
                })
        if hasattr(self, 'map_manager') and not runtime.config.active:
            self.map_manager._save_to_json()
        return time_best,self.solution

    def eval_calc(self,x,s:List[int],print_flag=False,name:str=''):
        dp=s[0]
        pp=s[1]
        cp=s[2]
        ulyp=s[3]
        tp=s[4]
        sp=s[5]
        ep=s[6]
        mbs=s[7]
        num_mb=s[8]
        result:float=0.0
        abc={'weight_size':self.optimizer.weight_size,'gradient_size':self.optimizer.gradient_size,'optimizer_state_size':self.optimizer.optimizer_state_size,
                                'head': self.head, 'b':mbs, 'hidden': self.hidden, 'vocab': self.vocab, 'layer': self.layer,
                                                                                'h_4': self.hidden*4, 'h_ffn': self.h_ffn,'h_moe': self.h_moe, 'h_share_moe': self.h_share_moe,'seq': self.seq, 'gbs': self.gbs, 'group': self.group,
                                                                                'dp':dp,'pp':pp,'cp':cp,'ulyp':ulyp,'tp':tp,'sp':sp,'ep':ep,
                                                                                'share_experts':self.share_experts,'Topk':self.Topk,'route_experts':self.experts,'capacity':self.capacity,
                                                                                'q_lora_rank': self.q_lora_rank,'kv_lora_rank':self.kv_lora_rank,'v_dim':self.v_head_dim,
                                                                                'qk_rope_dim':self.qk_rope_head_dim,'qk_nope_dim':self.qk_nope_head_dim}
        try:
            result=eval(x, globals(),abc)
        except Exception as e:
            raise RuntimeError(f'Expression evaluation failed: {x}\nBindings: {abc}') from e
        if print_flag:
            print(abc)
            print(f'{name}:{x}')
            print(f'{name}_result:{result}={result/1024/1024}MB={result/1024/1024/1024}GB')  
        return result

    def _tpds_profile_value(self, calc_name, calc_key, measure_fn):
        """Unified operator-profile lookup used by all TPDS variants."""
        runtime = TPDS_RUNTIME
        if runtime.config.active:
            return runtime.profile_value(self, calc_name, calc_key, measure_fn)
        self.sonmap=self.map_manager.data_map.setdefault(calc_name,{})
        self.value=self.sonmap.setdefault(calc_key,0)
        if self.value==0:
            self.value=measure_fn()
            self.sonmap[calc_key]=self.value
        return self.value

    def costmodel_timecost(self,s:List[int],cost_model:List[dict],auto_flag):
        #有没有auto_flag,cost_model的格式完全不同！
        #[4,2,1,1,1,1,1,8,8]
        def flops_calc(idx,s:List[int],moe_flag=False):
            idx_f=0.0
            if idx['calc']!= '0':
                if self.profile_calc_Flag and idx["left-matrix"] !=[]:
                    left_matrix=copy.deepcopy(idx["left-matrix"])
                    right_matrix=copy.deepcopy(idx["right-matrix"])
                    if moe_flag:
                        left_matrix[-2]=f'(capacity*Topk*ep/route_experts)*{left_matrix[-2]}'
                    left_list=[]
                    right_list=[]       
                    for left in left_matrix: 
                        left_list.append(self.eval_calc(left,s))
                    if right_matrix !=[]:
                        for right in right_matrix:
                            right_list.append(self.eval_calc(right,s))

                    tup_left=tuple(int(x) for x in left_list)
                    tup_right=tuple(int(x) for x in right_list)
                    calc_name=''
                    calc_key=''
                    #try:
                    if "layernorm" in idx["name"]:
                        calc_name=f"{self.device_name}-layernorm"
                        calc_key=str(tup_left)
                        idx_f=self._tpds_profile_value(calc_name,calc_key,lambda: Calc_LayerNorm(tup_left,self.hidden))
                    elif "rmsnorm" in idx["name"]:
                        calc_name=f"{self.device_name}-rmsnorm"
                        calc_key=str(tup_left)
                        idx_f=self._tpds_profile_value(calc_name,calc_key,lambda: Calc_RmsNorm(tup_left,tup_left[-1]))
                    elif "softmax" in idx["name"]:
                        calc_name=f"{self.device_name}-softmax"
                        calc_key=str(tup_left)
                        idx_f=self._tpds_profile_value(calc_name,calc_key,lambda: Calc_Softmax(tup_left))
                    elif "matmul" in idx["name"]:
                        if 'Flash_Attention' in idx['name'] and ('matmul-score' in idx['name'] or 'matmul-o' in idx['name']):
                            if 'matmul-score' in idx['name']:
                                self.q_flash=tup_left
                                self.k_flash=tup_right
                                idx_f=0
                            else:
                                self.v_flash=tup_right
                                calc_name=f"{self.device_name}-SDPA"
                                calc_key=str(self.q_flash+self.k_flash+self.v_flash)
                                idx_f=self._tpds_profile_value(calc_name,calc_key,lambda: Calc_SDPA(self.q_flash,self.k_flash,self.v_flash))
                        else:
                            calc_name=f"{self.device_name}-matmul"
                            calc_key=str(tup_left+tup_right)
                            idx_f=self._tpds_profile_value(calc_name,calc_key,lambda: Calc_Matmul(tup_left,tup_right))
                    elif "dropout" in idx["name"]:
                        calc_name=f"{self.device_name}-dropout"
                        calc_key=str(tup_left)
                        idx_f=self._tpds_profile_value(calc_name,calc_key,lambda: Calc_Dropout(tup_left,0.0))
                    elif "mask" in idx["name"]:
                        calc_name=f"{self.device_name}-mask"
                        calc_key=str(tup_left)
                        idx_f=self._tpds_profile_value(calc_name,calc_key,lambda: Calc_Mask(tup_left,tup_right))
                    elif "sub" in idx["name"]:
                        calc_name=f"{self.device_name}-sub"
                        calc_key=str(tup_left+tup_left)
                        idx_f=self._tpds_profile_value(calc_name,calc_key,lambda: Calc_Sub(tup_left,tup_left))
                    elif "mul" in idx["name"]:
                        calc_name=f"{self.device_name}-mul"
                        calc_key=str(tup_left+tup_left)
                        base_value=self._tpds_profile_value(calc_name,calc_key,lambda: Calc_Mul(tup_left,tup_left))
                        idx_f=base_value*(2*self.Topk+1) if "mul-experts" in idx["name"] else base_value
                    else:
                        idx_f=0
                    # except Exception as e:
                    #     print(f'calc_profile error,name is {calc_name}, num_shape is left:{tup_left},right:{tup_right}')
                    #     calc_part=f"({idx['calc']}*1e6)/{self.flops}"
                    #     self.value=self.eval_calc(calc_part,s)
                    #     idx_f=self.value
                        #import pdb; pdb.set_trace() 
                else:
                    calc_part=f"({idx['calc']}*1e6)/{self.flops}"#μs
                    idx_f=self.eval_calc(calc_part,s)
            return idx_f
        def comm_calc(idx,s:List[int],moe_flag=False):
            max_f:float=0.0
            max_of:float=0.0
            max_b:float=0.0
            max_ob:float=0.0
            temp_f:float=0.0
            temp_b:float=0.0
            for i in range(len(idx['com_data'])):
                cflist=[]
                pflist=[]
                cblist=[]
                pblist=[]
                if ',' in idx['com_data'][i][0] and ',' in idx['primitive'][i][0] and ',' in idx['com_data'][i][1] and ',' in idx['primitive'][i][1]:
                    cflist=idx['com_data'][i][0].split(',')
                    pflist=idx['primitive'][i][0].split(',')
                    cblist=idx['com_data'][i][1].split(',')
                    pblist=idx['primitive'][i][1].split(',')
                else:
                    cflist.append(idx['com_data'][i][0])
                    pflist.append(idx['primitive'][i][0])
                    cblist.append(idx['com_data'][i][1])
                    pblist.append(idx['primitive'][i][1])
                comm_f=0.0
                comm_b=0.0
                comm_of=0.0
                comm_ob=0.0
                for j in range(len(cflist)):
                    com_dataf=cflist[j]
                    com_datab=cblist[j]
                    primitivef=pflist[j]
                    primitiveb=pblist[j]
                    if moe_flag:
                       com_dataf=f'(capacity*Topk*ep/route_experts)*{com_dataf}' 
                       com_datab=f'(capacity*Topk*ep/route_experts)*{com_datab}'
                    if self.profile_comm_Flag:
                        partf=self.eval_calc(f"{SIZE_BF16}*{com_dataf}",s)
                        partb=self.eval_calc(f"{SIZE_BF16}*{com_datab}",s)
                        op_size_f=[primitivef,partf]
                        op_size_b=[primitiveb,partb]
                        temp_f=self.comm_data_processor.data_predict(op_size_f)
                        temp_b=self.comm_data_processor.data_predict(op_size_b)
                    else:
                        partf=f"({SIZE_BF16}*{com_dataf}*1e6)/{BAND_WIDTH_UNIDIRECTIONAL}"#μs
                        partb=f"({SIZE_BF16}*{com_datab}*1e6)/{BAND_WIDTH_UNIDIRECTIONAL}"#μs
                        #if 2 == len(idx['com_data'][i]):
                        temp_f=self.eval_calc(partf, s)                              
                        temp_b=self.eval_calc(partb, s)                          
                    if idx.get('comm_output', -1) >=0 and idx["primitive"][idx.get('comm_output', -1)][0] == primitivef:
                        comm_of=comm_of+temp_f
                        comm_ob=comm_ob+temp_b
                    else:
                        comm_f=comm_f+temp_f
                        comm_b=comm_b+temp_b
                if max_f < comm_f : max_f=comm_f
                if max_of < comm_of : max_of=comm_of
                if max_b < comm_b : max_b=comm_b
                if max_ob < comm_ob : max_ob=comm_ob
            return max_f,max_b,max_of,max_ob

        def event_base_cost(idx, s):
            """Cost of one local execution event before the recomputation replay.

            The key intentionally excludes idx['recompute']; recomputation changes
            the replay/schedule fact, not the forward operator/profile itself.
            Therefore a recomputation-only strategy delta can reuse the unchanged
            local event evidence and add replay cost afterward.
            """
            runtime = TPDS_RUNTIME
            key_payload = {
                'model': (self.layer, self.seq, self.hidden, self.head, self.group, self.h_ffn, self.experts, self.device_name),
                's': tuple(s),
                'name': idx.get('name'),
                'parallel': idx.get('parallel'),
                'parallel_mode': idx.get('parallel_mode'),
                'left': idx.get('left-matrix'),
                'right': idx.get('right-matrix'),
                'values': idx.get('values'),
                'calc': idx.get('calc'),
                'primitive': idx.get('primitive'),
                'com_data': idx.get('com_data'),
                'comm_output': idx.get('comm_output', -1),
                'to_memory': idx.get('to_memory'),
            }
            event_key = json.dumps(_jsonable(key_payload), sort_keys=True, separators=(',', ':'))
            if runtime.config.active and runtime.config.use_incremental and event_key in runtime.event_cost_cache:
                runtime.analysis('local_event', cache_hit=True)
                return runtime.event_cost_cache[event_key]

            et0 = time.perf_counter()
            idx_left=idx_right=idx_output=0.0
            idx_left_m=idx_right_m=idx_output_m=0.0
            if "aclu" not in idx["name"]:
                calc_part_left=f"({SIZE_BF16}*{'*'.join(idx['left-matrix'])}*1e6)/{BAND_WIDTH_MEMORY_TRANS}"
                calc_part_right=f"({SIZE_BF16}*{'*'.join(idx['right-matrix'])}*1e6)/{BAND_WIDTH_MEMORY_TRANS}"
                if idx['right-matrix']=='':
                    calc_part_right='0'
                if 'Flash_Attention' in idx['name']:
                    if 'matmul-o' in idx['name']:
                        calc_part_left='0'
                        calc_part_right='0'
                    elif 'matmul-score' in idx['name']:
                        temp_left='*'.join(idx['left-matrix'])
                        temp_result=self.eval_calc(temp_left,s)
                        if temp_result < 2097152:
                            calc_part_left=f"({SIZE_BF16}*2097152*1e6)/{BAND_WIDTH_MEMORY_TRANS}"
                        calc_part_left=f'50 * {calc_part_left}'
                idx_left=self.eval_calc(calc_part_left,s)
                idx_right=self.eval_calc(calc_part_right,s)
            if not idx["to_memory"] =='0':
                # Output-write traffic depends on the actual materialized output size.
                # The legacy evaluator accidentally evaluated calc_part_right here;
                # that both double-counted the right-input read for some operators and
                # left calc_part_right undefined for output-producing ACLU events.
                idx_output_expr=f"({SIZE_BF16}*{idx['to_memory']}*1e6)/{BAND_WIDTH_MEMORY_TRANS}"
                idx_output=self.eval_calc(idx_output_expr,s)
            if 'moe' in idx["name"]:
                idx_left_m=(self.capacity*self.Topk*ep/self.experts)*idx_left
                idx_right_m=(self.capacity*self.Topk*ep/self.experts)*idx_right
                idx_output_m=(self.capacity*self.Topk*ep/self.experts)*idx_output
            idx_f=flops_calc(idx,s)
            idx_f_m=0.0
            if 'moe' in idx['name'] and 'mul-experts' not in idx['name']:
                idx_f_m=flops_calc(idx,s,moe_flag=True)
            max_f,max_b,max_of,max_ob=comm_calc(idx,s)
            max_f_m=max_b_m=max_of_m=max_ob_m=0.0
            comm_coef=5.0
            if 'index-comm' in idx["name"] and alltoall_overlap_flag:
                max_f=0.0
                max_of=0.0
            if 'moe' in idx['name'] and 'ep' not in idx["parallel"]:
                max_f_m,max_b_m,max_of_m,max_ob_m=comm_calc(idx,s,moe_flag=True)
            base_f=idx_left + idx_right + comm_coef*max_f + idx_f + comm_coef*max_of + idx_output
            base_b=comm_coef*max_b + 2*idx_f + comm_coef*max_ob
            if 'moe' in idx['name']:
                base_f=base_f+idx_left_m+idx_right_m+comm_coef*max_f_m+idx_f_m+comm_coef*max_of_m+idx_output_m
                base_b=base_b+comm_coef*max_b_m+2*idx_f_m+comm_coef*max_ob_m
            if runtime.config.active:
                runtime.analysis('local_event', time.perf_counter()-et0)
                if runtime.config.use_incremental:
                    runtime.event_cost_cache[event_key]=(base_f,base_b)
            return base_f,base_b

        result_f:float=0.0
        result_b:float=0.0
        result:float=0.0
        sum_f:float=0.0
        sum_b:float=0.0 
        dp=s[0]
        pp=s[1]
        cp=s[2]
        ulyp=s[3]
        tp=s[4]
        sp=s[5]
        ep=s[6]
        mbs=s[7]
        num_mb=s[8]
        inter_overlap=1.0#算子内的计算-通信重叠率。1为完全重叠，0为完全独立
        stage=[]
        pipeline=[]
        pipeline_waittime=[0.0 for _ in range(num_mb)]
        vppnum=1
        alltoall_overlap_flag=False
        moe_tp_extend_ep=getattr(self.args, 'moe_tp_extend_ep', False)
        moe_extended_tp =getattr(self.args, 'moe_extended_tp', False)
        expert_tensor_parallel_size =getattr(self.args, 'expert_tensor_parallel_size', None)
        if (tp==1 or moe_tp_extend_ep) or (not moe_extended_tp and expert_tensor_parallel_size is None):#此时默认开了moe_alltoall_overlap_comm和moe_permutation_async_comm
            alltoall_overlap_flag=True
        for opti in self.included_parallels.pipe_p.optimizations:
                if 'VirtualPipe' == opti.instance_name and pp > 1 :
                    vppnum=opti.get_vppnum(pp)      
        if auto_flag:
            for d in cost_model:#后面要考虑Tensor耗时的时候，可以在d里加上
                result_f=eval('+'.join(d),globals(),{'b': mbs})
                result_b=2*result_f
                stage.append([result_f,result_b])#前向总耗时，后向总耗时 
                #pipeline_waittime=self.pipeline_time_auto
        else:
            if pp > 1 :
                list_costmodel=split_with_offset(cost_model[1:len(cost_model)-1],pp*vppnum)
                #list_costmodel=np.array_split(cost_model[1:len(cost_model)-1],pp*vppnum)
                list_costmodel[0]=[cost_model[0]]+list(list_costmodel[0])
                list_costmodel[-1]=list(list_costmodel[-1])+[cost_model[-1]]
                #list_costmodel[0]=np.append([cost_model[0]],[list_costmodel[0]])
                #list_costmodel[-1]=np.append([list_costmodel[-1]],[cost_model[-1]])
            else:
                list_costmodel=[cost_model] 
            for d_list in list_costmodel:
                d = [item for sublist in d_list for item in sublist]
                result_f=result_b=0.0
                #计算stage 时间（正+反）
                #在stage内，通讯和计算都是交替且细碎的，因此存在相互隐藏的优化空间（字节的工作）。将stage内所有的通讯和计算按顺序走完，并将时间相加，则得到stage的时间。
                for idx in d:
                    sum_f, sum_b = event_base_cost(idx, s)
                    if idx['recompute']:
                        sum_b += sum_f
                    result_f += sum_f
                    result_b += sum_b
                stage.append([result_f,result_b])#前向总耗时，后向总耗时
        if pp == 1:
            result=(result_f + result_b) * num_mb
        # if pp==1 and tp == 8 and num_mb==128:
        #     import pdb; pdb.set_trace() 
        else:
            if vppnum > 1:
                num_mb=num_mb*vppnum
                pipeline:List[PipeStage] = [PipeStage(idx=d, pp=pp, vppnum=vppnum,stage=stage,acti_form='(pp*vpp)+pp-1-(2*i)') for d in range(pp)]
            else:
                pipeline: List[PipeStage] = [PipeStage(idx=d, pp=pp, stage=stage) for d in range(pp)]
                 
            pipeline[-1].bf_flag=False#warmup完成后再开启
            if len(pipeline_waittime) >= num_mb:
                pipeline_waittime=pipeline_waittime[0:num_mb]
            else:
                for _ in range(num_mb-len(pipeline_waittime)):
                    pipeline_waittime.append(pipeline_waittime[-1])
            b_status: List[float] = [0.0 for _ in range(num_mb)]
            pipeline[0].f_list = [d for d in range(num_mb)]
            pipeline[0].f_time = [0.0 for _ in range(num_mb)]
            for idx in pipeline:
                idx.pipeline_waittime=[0.0 for _ in range(num_mb)]
            pipeline[0].pipeline_waittime=pipeline_waittime
            # if pp==4 and tp == 1 and num_mb==64:
            #     import pdb; pdb.set_trace() 
            tip = 0
            while b_status[-1] == 0.0:
                for d in pipeline:
                    if d.f_list or d.b_list:
                        d.flag = False #本回合休息
                for i in range(tip, num_mb):  
                    if False not in [d.flag for d in pipeline]:#所有玩家都完成了本回合
                        break
                    if 0.0 != b_status[i]:  # 已完成
                        tip = i + 1
                        continue
                    #pp之间的通讯为点到点通信，数据大小为b * seq * hidden，因为和stage的耗时不是一个量级的，因此忽略。（PP通讯必定被计算掩盖）
                    for j in range(pp):  # 查看当前batch走到了哪里
                        if not pipeline[j].flag and pipeline[j].f_list and i == pipeline[j].f_list[0] and pipeline[j].num_acti < pipeline[j].num_acti_max:  # 看看在不在前向计算队列里
                            pipeline[j].status[0] = pipeline[j].f_list.pop(0)
                            pipeline[j].status[1] = pipeline[j].pipeline_waittime[i]+pipeline[j].get_time_f(i) + max(pipeline[j].f_time.pop(0), pipeline[j].status[1])
                            if j == pp - 1:
                                pipeline[j].b_list.append(pipeline[j].status[0])
                                pipeline[j].b_time.append(pipeline[j].status[1])
                            else:
                                pipeline[j + 1].f_list.append(pipeline[j].status[0])
                                pipeline[j + 1].f_time.append(pipeline[j].status[1])
                            pipeline[j].num_acti += 1
                            pipeline[j].flag = True
                        #确保最后一层等warmup完成后才开始反向
                        if not pipeline[j].bf_flag and j == pp -1 and pipeline[j].num_acti==pipeline[j].num_acti_max:                          
                            pipeline[j].bf_flag=True
                        if pipeline[j].bf_flag and not pipeline[j].flag and pipeline[j].b_list and i == pipeline[j].b_list[0]:
                            pipeline[j].status[0] = pipeline[j].b_list.pop(0)
                            pipeline[j].status[1] = pipeline[j].get_time_b(i) + max(pipeline[j].b_time.pop(0), pipeline[j].status[1])
                            if j == 0:
                                b_status[i] = pipeline[j].status[1]
                            else:
                                pipeline[j - 1].b_list.append(pipeline[j].status[0])
                                pipeline[j - 1].b_time.append(pipeline[j].status[1])
                            pipeline[j].num_acti -= 1
                            pipeline[j].flag = True

            result = b_status[-1]
        self.calc_count=self.calc_count+1
        if self.print_flag:
            print(f"finish:{s}")
        # print(f"finish_num:{self.calc_count}")               
        return result


    def calculate_compute_power(self) -> float:
        """
        算力开销计算
        返回算力开销（单位：FLOPs）
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def set_optimizer_params(self, optimizer_name: str):
        """
        设置子类的优化器参数为指定优化器的参数
        :param optimizer_name: 优化器名称（如 "Adam"）
        """
        if optimizer_name in self.optimizer_options:
            self.optimizer = self.optimizer_options[optimizer_name]
        else:
            raise ValueError(f"Optimizer '{optimizer_name}' not found in optimizer_options")       

@dataclass    
class Embedding(ModuleConfig):
    """
    Embedding模块类，继承自ModuleConfig
    """
    def __init__(self,args,instance_name:str= "embedding",optimizations:List=None):

        # 调用父类初始化
        super().__init__(args,instance_name=instance_name)

        # 计算输入/输出形状
        self.input_shape=['4*b',self.seq] #刚输入时是fp32类型
        #自身参数
        self.W= [{"name": "Wvocab", "shape": ["vocab","hidden"]}]

        self.calc_module({"type": f"index-embedding-{self.W[0]['name']}",
                           "left-matrix":['4*b' , self.seq , '1'],
                           "right-matrix":[self.W[0]['shape'][0] , self.W[0]['shape'][1]], 
                          "output-matrix":['b' , self.seq , self.W[0]['shape'][1]],
                          "calc":"0",#通过对离散数据进行查表索引，将离散数据变成向量，从而完成了升维
                          "parallel":"tp",
                          "parallel_mode":"row"},ioput='input')#存input
        #hidden_dropout不影响embedding层的dropout
        self.calc_module({"type": "dropout-h",
                            "left-matrix":['b' , self.seq , self.W[0]['shape'][1]],
                            "right-matrix":['b' , self.seq , self.W[0]['shape'][1]], 
                            "output-matrix":['b' , self.seq , self.W[0]['shape'][1]],
                            "calc":f"b * {self.seq} * {self.hidden}",
                            "parallel":"sp"},flag=True)#存dropout
        
        self.output_shape = ['b',self.seq,self.hidden] 

        #self.activation_list=["input","dropout-h"]#激活值统一存储left,即输入，并新增right_flag，可以存right
        #是否启用优化
        if optimizations != None:
            self.apply_optimizations(optimizations)

        self.calculate_flow()

@dataclass
class Output(ModuleConfig):
    """
    Output模块类，继承自ModuleConfig
    """ 
    def __init__(self, args,instance_name: str = "output",optimizations:List=None):
        """
        初始化Output模块
        """
        # 调用父类初始化
        super().__init__(args,instance_name=instance_name)

        # 计算输入/输出形状
        self.input_shape = ['b',self.seq,self.hidden]
        
        #自身参数
        self.W= [{"name": "Wvocab", "shape":["hidden","vocab"]}]
        
        self.calc_module({"type": f"matmul-{self.W[0]['name']}",
                           "left-matrix":['b' , self.seq , self.hidden],
                           "right-matrix":[self.W[0]['shape'][0] , self.W[0]['shape'][1]],#已经过转置（T）
                          "output-matrix":['b' , self.seq , self.W[0]['shape'][1]],#转成了fb32类型，所以在b这里*2（特殊处理）
                          "calc":f"b*{self.seq}*2*{self.hidden}*{self.W[0]['shape'][1]}",
                          "parallel":"tp",
                          "parallel_mode":"col"},flag=True,ioput=self.ioput)
        #后两层是softmax和损失计算，现在大多都合并，但是我们这里需要对计算过程的内存分配进行评估，所以拆开
        #后面两层实际上不会有激活值表示，因为算完后直接开始反向，反向完就释放了
        #但是在计算的过程中，依旧需要分配剩余内存，会在现有参数+激活上再给压力，因此也参与显存计算
        #softmax归一化
        self.calc_module({"type": f"softmax-Outlayer",
                           "left-matrix":['2*b' , self.seq , self.W[0]['shape'][1]],#转成了fb32类型，所以在b这里*2（特殊处理）
                           "right-matrix":"",
                          "output-matrix":['2*b' , self.seq , self.W[0]['shape'][1]],
                          "calc":f"17* 2*b * {self.seq} * {self.W[0]['shape'][1]}",
                          "parallel":"tpf"},flag=True)
        #损失计算:归一化后的[b,s,v]代表每个词的词表内的概率，对上真实输入[b,s]和当前v得到的[b,s,v]
        #相减就得到了最终的损失值。得到损失值后马上进行反向
        #因此这一步的结果只会储存很短一段时间(大概50ms),但依然需要内存分配
        self.calc_module({"type": f"sub-Outlayer",
                           "left-matrix":['2*b' , self.seq , self.W[0]['shape'][1]],
                           "right-matrix":['2*b' , self.seq , self.W[0]['shape'][1]],
                          "output-matrix":['2*b' , self.seq , self.W[0]['shape'][1]],
                          "calc":f"2*b * {self.seq} * {self.W[0]['shape'][1]}",
                          "parallel":"tpf"},flag=True)
        self.output_shape = ['b',self.seq,self.hidden]

        #self.activation_list=["input","output"]

        #是否启用优化
        if optimizations != None:
            self.apply_optimizations(optimizations)

        self.calculate_flow()
      
@dataclass
class LayerNorm(ModuleConfig):
    """
    LayerNorm模块类，继承自ModuleConfig
    """
    def __init__(self, args,instance_name: str = "layer_norm",optimizations:List=None):
        # 调用父类初始化
        super().__init__(args,instance_name=instance_name)
        # 计算输入/输出形状
        self.input_shape = ['b' ,self.seq , self.hidden]
        #自身参数
        self.W= [{"name": "Wmean", "shape": ["hidden"]},
                 {"name": "Wsquare", "shape": ["hidden"]}] 
        self.calc_module({"type": "layernorm",
                           "left-matrix":['b' , self.seq , self.hidden],
                           "right-matrix":"", 
                          "output-matrix":['b' , self.seq , self.hidden],
                          "calc":f"8 * b * {self.seq} * {self.hidden}",#根据LayerNorm公式推导出的近似浮点计算数
                          "parallel":"sp"},ioput='input')
        
        self.output_shape = ['b' ,self.seq , self.hidden]
        #self.activation_list=["input"]

        #是否启用优化
        if optimizations != None:
            self.apply_optimizations(optimizations)

        self.calculate_flow()
        
@dataclass
class RMS_Norm(ModuleConfig):
    """
    RMS_Norm模块类，继承自ModuleConfig
    """
    def __init__(self, args,instance_name: str = "RMS_norm",optimizations:List=None):    
        # 调用父类初始化
        super().__init__(args,instance_name=instance_name)
        # 计算输入/输出形状
        self.input_shape = ['b' ,self.seq , self.hidden]
        #自身参数
        self.W= [{"name": "Wrms", "shape": ["hidden"]}]
        self.calc_module({"type": "rmsnorm",
                           "left-matrix":['b' , self.seq , self.hidden],
                           "right-matrix":"", 
                          "output-matrix":['b' , self.seq , self.hidden],
                          "calc":f"5 * b * {self.seq} * {self.hidden}",#根据RMSNorm公式推导出的近似浮点计算数
                          "parallel":"sp"},ioput='input')
        self.output_shape = ['b' ,self.seq , self.hidden]    
        #是否启用优化
        if optimizations != None:
            self.apply_optimizations(optimizations)
        self.calculate_flow()

@dataclass
class MLA(ModuleConfig):
    """
    Self_attention模块类，继承自ModuleConfig
    """
    def __init__(self, args,instance_name: str = "mla",optimizations:List=None):
        """
        初始化Self_attention模块
        """
        # 调用父类初始化
        super().__init__(args,instance_name=instance_name)

        # 计算输入/输出形状
        self.input_shape = ['b' ,self.seq , self.hidden]      

        #自身参数
        self.W= [{"name": "Wqkv", "shape": ["hidden", "(kv_lora_rank + qk_rope_dim + qk_nope_dim*head + qk_rope_dim*head)"]},
                  {"name": "Wkvup", "shape": ["kv_lora_rank", "qk_nope_dim * head * 2"]},
                  {"name": "Wproj", "shape": ["v_dim * head", "hidden"]}]
        if args.qk_layernorm:
            self.W.append({"name": "qk_rmsnorm", "shape": ["kv_lora_rank"]})

        self.calc_module({"type": f"matmul-{self.W[0]['name']}",#qkv操作
                           "left-matrix":['b' , self.seq , self.hidden],
                           "right-matrix":[self.W[0]['shape'][0] , self.W[0]['shape'][1]], 
                          "output-matrix":['b' , self.seq , self.W[0]['shape'][1]],
                          "calc":f"b*{self.seq}*{self.W[0]['shape'][1]}*2*{self.W[0]['shape'][0]}",
                          "parallel":""},True,ioput=self.ioput)
        qkv_list=self.W[0]['shape'][1].split("+")
        kvdown=f'{qkv_list[0]})'
        kr=qkv_list[1]
        qn=qkv_list[2]
        qr=f'({qkv_list[3]}'
        if args.qk_layernorm:
             self.calc_module({"type": "rmsnorm-kv",#kvdown
                           "left-matrix":['b' , self.seq , f"{kvdown}"],
                           "right-matrix":"", 
                          "output-matrix":['b' , self.seq , f"{kvdown}"],
                          "calc":f"5 * b * {self.seq} * {kvdown}",
                          "parallel":""},True)
        self.calc_module({"type": f"matmul-{self.W[1]['name']}",#kvup
                           "left-matrix":['b' , self.seq , f'{kvdown}'],
                           "right-matrix":[self.W[1]['shape'][0] , self.W[1]['shape'][1]], 
                          "output-matrix":['b' , self.seq , self.W[1]['shape'][1]],
                          "calc":f"b*{self.seq}*{self.W[1]['shape'][1]}*2*{self.W[1]['shape'][0]}",
                          "parallel":""},True)
        #接下来进入flash_attention阶段，先写死吧
        self.calc_module({"type": "matmul-score",#q*k
                           "left-matrix":['b' , 'head' , self.seq , f'{qn}/head*2'],#qkv在这个维度，全部扩容成了qn*2的大小
                           "right-matrix":['b' , 'head' , f'{qn}/head*2' , self.seq], #kr显性复制head倍，再与['b' , self.seq , self.W[3]['shape'][1]/2]相加
                          "output-matrix":['b' , 'head' , self.seq , self.seq],
                          "calc":f"b * head * {self.seq} * {self.seq} *2*{qn}*2",
                          "parallel":"ulypf,cp,tpf"},True)
        self.calc_module({"type": "mask",
                           "left-matrix":['b' , 'head' , self.seq , self.seq],
                           "right-matrix":"", 
                          "output-matrix":['b' , 'head' , self.seq , self.seq],
                          "calc":f"b * head * {self.seq} * {self.seq}",
                          "parallel":"ulypf,tpf"},True) 
        self.calc_module({"type": "softmax",
                           "left-matrix":['b' , 'head' , self.seq , self.seq],
                           "right-matrix":"", 
                          "output-matrix":['b' , 'head' , self.seq , self.seq],
                          "calc":f"17*b * head * {self.seq} * {self.seq}",
                          "parallel":"ulypf,tpf"},True)#若向量N=1000，则FLOPs = 1000×15（指数）+999（求和）+1000（归一化）=16,999 ≈ 17K。
        if self.attention_dropout > 0.0:
            self.calc_module({"type": "dropout-a",
                               "left-matrix":['b' , 'head' , self.seq , self.seq],
                               "right-matrix":"", 
                          "output-matrix":['b' , 'head' , self.seq , self.seq],
                          "calc":f"b * head * {self.seq} * {self.seq}",
                          "parallel":"ulypf,tpf"},True)
        self.calc_module({"type": "matmul-o",
                           "left-matrix":['b' , 'head' , self.seq , self.seq],
                           "right-matrix":['b' , 'head' , self.seq, f'{qn}/head'], 
                          "output-matrix":['b' , 'head' , self.seq, f'{qn}/head'],#a是原注意头，self.W[5]['shape'][1]]是通过参数指定扩维生成的
                          "calc":f"b * head * {self.seq} * {qn}/head * 2*{self.seq}",
                          "parallel":""},True)
        
        self.calc_module({"type": f"matmul-{self.W[2]['name']}",
                            "left-matrix":['b' , self.seq, f'{qn}'],
                            "right-matrix":[self.W[2]['shape'][0] , self.W[2]['shape'][1]], 
                            "output-matrix":['b' , self.seq , self.W[2]['shape'][1]],
                            "calc":f"b*{self.seq}*{self.W[2]['shape'][1]}*2*{self.W[2]['shape'][0]}",
                            "parallel":""},True)#残差链接,这一步之后才是输出
        if self.hidden_dropout > 0.0:
            self.calc_module({"type": "dropout-h",
                                    "left-matrix":['b' , self.seq , self.W[2]['shape'][1]],
                                    "right-matrix":['b' , self.seq , self.W[2]['shape'][1]],
                                    "output-matrix":['b' , self.seq , self.W[2]['shape'][1]],
                                    "calc":f"b * {self.seq} * {self.W[2]['shape'][1]}",
                                    "parallel":""},True) 
        self.output_shape = ['b',self.seq,self.hidden]
        
        #是否启用优化
        if optimizations != []:
            self.apply_optimizations(optimizations)
        if args.add_qkv_bias:
            self.use_bias=True
        self.calculate_flow()#得到激活，通信和计算

        # self.activation_values.append({"type": "weight-input", "values": f"b * s * h","parallel":"cp,ulyp,tp","parallel_mode":"col"})#输入


class Self_attention(ModuleConfig):
    """
    Self_attention模块类，继承自ModuleConfig
    """
    def __init__(self, args,instance_name: str = "self_attention",optimizations:List=None):
        """
        初始化Self_attention模块
        """
        # 调用父类初始化
        super().__init__(args,instance_name=instance_name)

        # 计算输入/输出形状
        self.input_shape = ['b' ,self.seq , self.hidden]      
        
        #在qkv之后，会变成kv_channels和head的乘积，之后再变成hidden的大小，所以中间的变化用一个比率来控制

        #自身参数
        self.W= [{"name": "Wqkv", "shape": ["hidden", f"(head*{args.kv_channels}+group*{args.kv_channels}+group*{args.kv_channels})"]},
                {"name": "Wproj", "shape": [f"({args.kv_channels}*head)", "hidden"]}]
        
        #定义算子流程
        tp_cut=[['head','group'],None]         
        self.calc_module({"type": f"matmul-{self.W[0]['name']}",
                           "left-matrix":['b' , self.seq , self.hidden],
                           "right-matrix":[self.W[0]['shape'][0] , self.W[0]['shape'][1]], 
                          "output-matrix":['b' , self.seq , self.W[0]['shape'][1]],
                          "calc":f"b*{self.seq}*2*{self.W[0]['shape'][0]}*{self.W[0]['shape'][1]}",
                          "parallel":"ulyp,tp",#为了方便，把ulyp放到这
                          "parallel_mode":"col"},True,ioput=self.ioput,tp_cut=tp_cut)
        weight_list=self.W[0]['shape'][1].split('+')
        weight_q=f"{weight_list[0]})"
        weight_k=f"({weight_list[1]})"
        weight_v=f"({weight_list[2]}"
        if args.qk_layernorm:
             self.calc_module({"type": "q-rmsnorm",
                           "left-matrix":['b' , self.seq , weight_q],
                           "right-matrix":"", 
                          "output-matrix":['b', self.seq , weight_q],
                          "calc":f"5 * b * {self.seq} * {weight_q}",
                          "parallel":"ulypf,tpf"},True)
             self.calc_module({"type": "k-rmsnorm",
                           "left-matrix":['b' , self.seq , weight_k],
                           "right-matrix":"", 
                          "output-matrix":['b', self.seq , weight_k],
                          "calc":f"5 * b * {self.seq} * {weight_k}",
                          "parallel":"ulypf,tpf"},True)
        # self.calc_module({"type": f"matmul-{self.W[1]['name']}",
        #                    "left-matrix":['b' , self.seq , self.hidden],
        #                    "right-matrix":[self.W[1]['shape'][0] , self.W[1]['shape'][1]], 
        #                   "output-matrix":['b' , self.seq , self.W[1]['shape'][1]],#g=h/a,正常bsh*(bsh)T=bss。多了注意力a之后，变成basg
        #                   "calc":f"b*{self.seq}*2*{self.W[1]['shape'][0]}*{self.W[1]['shape'][1]}",
        #                   "parallel":"tpf"},True,tp_same=True)
        #self.calc_module(RoPE，calc=3*bsh)先不加，因为现在大多都是RoPE的融合算子
        self.calc_module({"type": "matmul-score",#q*k
                           "left-matrix":['b' , f"{weight_q}/{args.kv_channels}" , self.seq , f"{args.kv_channels}"],
                           "right-matrix":['b' , f"{weight_k}/{args.kv_channels}" , f"{args.kv_channels}" , self.seq], #g=h/a
                          "output-matrix":['b' , f"{weight_q}/{args.kv_channels}" , self.seq , self.seq],
                          "calc":f"b * {weight_q}/{args.kv_channels} * {self.seq} * {self.seq} *2*{args.kv_channels}",
                          "parallel":"ulypf,cp,tpf"},True)      
        self.calc_module({"type": "mask",
                           "left-matrix":['b' , f"{weight_q}/{args.kv_channels}" , self.seq , self.seq],
                           "right-matrix":"", 
                          "output-matrix":['b' , f"{weight_q}/{args.kv_channels}" , self.seq , self.seq],
                          "calc":f"b * {weight_q}/{args.kv_channels} * {self.seq} * {self.seq}",
                          "parallel":"ulypf,tpf"},True) 
        self.calc_module({"type": "softmax",
                           "left-matrix":['b' , f"{weight_q}/{args.kv_channels}" , self.seq , self.seq],
                           "right-matrix":"", 
                          "output-matrix":['b' , f"{weight_q}/{args.kv_channels}" , self.seq , self.seq],
                          "calc":f"17*b * {weight_q}/{args.kv_channels} * {self.seq} * {self.seq}",
                          "parallel":"ulypf,tpf"},True)#若向量N=1000，则FLOPs = 1000×15（指数）+999（求和）+1000（归一化）=16,999 ≈ 17K。
        if self.attention_dropout > 0.0:
            self.calc_module({"type": "dropout-a",
                               "left-matrix":['b' , f"{weight_q}/{args.kv_channels}" , self.seq , self.seq],
                               "right-matrix":"", 
                          "output-matrix":['b' , f"{weight_q}/{args.kv_channels}" , self.seq , self.seq],
                          "calc":f"b * {weight_q}/{args.kv_channels} * {self.seq} * {self.seq}",
                          "parallel":"ulypf,tpf"},True)
        # self.calc_module({"type": f"matmul-{self.W[2]['name']}",
        #                    "left-matrix":['b' , self.seq , self.hidden],
        #                    "right-matrix":[self.W[2]['shape'][0] , self.W[2]['shape'][1]], 
        #                   "output-matrix":['b' , self.seq, self.W[2]['shape'][1]],#g=h/a
        #                   "calc":f"b*{self.seq}*2*{self.W[2]['shape'][0]}*{self.W[2]['shape'][1]}",
        #                   "parallel":"tpf"},True,tp_same=True)#这里需要tp_same,来确保重复执行tp的切分，但是又不能用tp，因为qkv的tp通讯是同时的
        self.calc_module({"type": "matmul-o",
                           "left-matrix":['b' , f"{weight_q}/{args.kv_channels}" , self.seq , self.seq],
                           "right-matrix":['b' , f"{weight_v}/{args.kv_channels}" , self.seq, f"{args.kv_channels}"], #b* a * s * h/a
                          "output-matrix":['b' , f"{weight_q}/{args.kv_channels}" , self.seq, f"{args.kv_channels}"],
                          "calc":f"b*{weight_q}/{args.kv_channels}*{self.seq}*2*{self.seq}*{args.kv_channels}",
                          "parallel":"ulyp,cp,tpf"},True)
        tp_cut=[['head'],['head']]
        self.calc_module({"type": f"matmul-{self.W[1]['name']}",
                            "left-matrix":['b' , self.seq, weight_q],
                            "right-matrix":[self.W[1]['shape'][0] , self.W[1]['shape'][1]], 
                            "output-matrix":['b' , self.seq , self.W[1]['shape'][1]],
                            "calc":f"b*{self.seq}*2*{self.W[1]['shape'][0]}*{self.W[1]['shape'][1]}",
                            "parallel":"tp",
                            "parallel_mode":"row"})
        if self.hidden_dropout > 0.0:
            self.calc_module({"type": "dropout-h",
                                    "left-matrix":['b' , self.seq , self.W[1]['shape'][1]],
                                    "right-matrix":['b' , self.seq , self.W[1]['shape'][1]],
                                    "output-matrix":['b' , self.seq , self.W[1]['shape'][1]],
                                    "calc":f"b * {self.seq} * {self.W[1]['shape'][1]}",
                                    "parallel":"sp"},True) 
        self.output_shape = ['b',self.seq,self.hidden]
        
        #是否启用优化
        if optimizations != []:
            self.apply_optimizations(optimizations)
        if args.add_qkv_bias:
            self.use_bias=True
       
        self.calculate_flow()#得到激活，通信和计算

        # self.activation_values.append({"type": "weight-input", "values": f"b * s * h","parallel":"cp,ulyp,tp","parallel_mode":"col"})#输入

@dataclass
class ffn_relu(ModuleConfig):
    """
    ffn_relu模块类，继承自ModuleConfig
    """
    def __init__(self,args,flag_moe,instance_name: str = "mlp_ffn_relu",optimizations:List=None):
        """
        初始化ffn_relu模块
        """
        # 调用父类初始化
        super().__init__(args,flag_moe=flag_moe,instance_name=instance_name)
        # 计算输入/输出形状
        self.input_shape = ['b' ,self.seq , self.hidden]
        # 定义权重集合 W
        if self.flag_moe and args.num_experts is not None:
            self.W = [{"name": "W1", "shape": ["hidden", "h_moe"]},      # Wq = [h, 4h]
                  {"name": "W2", "shape": ["h_moe", "hidden"]},
                  {"name": "WGating_Network", "shape": ["hidden", "route_experts"]}]
        else:
            self.W = [{"name": "W1", "shape": ["hidden", "h_4"]},      # Wq = [h, 4h]
                  {"name": "W2", "shape": ["h_4", "hidden"]}]      # Wk = [4h, h]
                  # Wk = [4h, h]
        if self.flag_moe and args.num_experts is not None:
            self.calc_module({"type": f"matmul-{self.W[-1]['name']}",
                           "left-matrix":['b' , self.seq , self.hidden],
                           "right-matrix":[self.W[-1]['shape'][0] , self.W[-1]['shape'][1]], 
                          "output-matrix":['b',self.seq,self.W[-1]['shape'][1]],
                          "calc":f"b*{self.seq}*2*{self.hidden}*{self.W[-1]['shape'][1]}",
                          "parallel":"tp",
                          "parallel_mode":"row"},True)
            #if args.noisy_gate_policy is not None:#noisy_gate_policy=None、"RSample"（高斯噪声）或"Jitter"（均匀噪声）
            #  self.calc_module({"type": "Topkchoose",#负载均衡的噪音（未写好）
            #                "left-matrix":f"b*{self.seq}*{self.W[-1]['shape'][1]}",
            #                "right-matrix":f"{self.W[-1]['shape'][1]}* Topk", 
            #               "output-matrix":f"b*{self.seq}*Topk",#采用索引的方式，和embedding词表一样
            #               "calc":f"0",
            #               "parallel":"tpf"},True)
            self.calc_module({"type": "mask",#非TopK位置设为负无穷,仅TopK位置保留原分数
                           "left-matrix":['b',self.seq,self.W[-1]['shape'][1]],
                           "right-matrix":['b',self.seq,self.W[-1]['shape'][1]], 
                          "output-matrix":['b',self.seq,self.W[-1]['shape'][1]],#采用索引的方式，和embedding词表一样
                          "calc":f"b*{self.seq}*{self.W[-1]['shape'][1]}",
                          "parallel":"sp"},True)
            #moe_use_sinkhorn=False（未做，做的话得区分普通softmax和sinkhorn，且要在后端调用实现）
            #​标准Softmax路由​：直接按概率分配token，易导致负载倾斜。
            #Sinkhorn优化​：通过迭代归一化约束专家负载方差≤15%，但增加10%计算开销
            self.calc_module({"type": "softmax-routing",#路由，生成专家选择概率分布
                           "left-matrix":['b',self.seq,self.W[-1]['shape'][1]],
                           "right-matrix":"", 
                          "output-matrix":['b',self.seq,self.W[-1]['shape'][1]],
                          "calc":f"17*b*{self.seq}*{self.W[-1]['shape'][1]}",#这里采用32位，因为16位会不稳定
                          "parallel":""},True)#这里可以支持sp,但现有资料没有明确说明这个用到了sp
            self.calc_module({"type": "index-Topkchoose",#TopK选择
                           "left-matrix":['b',self.seq,self.W[-1]['shape'][1]],
                           "right-matrix":'', 
                          "output-matrix":['b',self.seq,'Topk'],#采用类似于冒泡索引的方式，和embedding词表一样，这里得到的是Topk的权重
                          "calc":f"0",
                          "parallel":""},True,acti=2)#存了指引和权重
            if args.moe_router_pre_softmax and self.Topk>1:
                self.calc_module({"type": "softmax-topk",#将权重归一化
                           "left-matrix":['b',self.seq,'Topk'],
                           "right-matrix":"", 
                          "output-matrix":['b',self.seq,'Topk'],
                          "calc":f"17*b*{self.seq}*Topk",
                          "parallel":""})
            self.calc_module({"type": "index-comm",#根据权重,用通信分发s
                           "left-matrix":['b',self.seq,'Topk'],
                           "right-matrix":"", 
                          "output-matrix":['b',self.seq,self.hidden],
                          "calc":f"0",
                          "parallel":"ep"},True)#每个节点需要保存自己的输入
            
        #根据topk里面的索引，将原b*s*h发送给专家，即通过门控系统，建立了一个h->k的索引，根据索引k将h给对应的k号专家
        #专家内的操作，最后要根据专家的个数确定具体的激活。所有专家的总激活=bsh*Topk（发给几个专家就代表Token被重复算了几次），
        # 但是在ep下，单节点内的（expert/ep）个专家的激活不确定，要看有多少个token被发送过来
        #单节点下，极端情况下，TopK的模型都在自己的节点上，即b*min(topk,expert/ep)*s*h。
        #理想情况下，即b*(s*topk)/ep*h。
        #单ffn下，极端情况是b*s*h。
        #理想情况是 b*(s/experts)*topk*h。
        #用理想*capacity进行测试，如果没有指定capacity，则capacity=1.25
        self.calc_module({"type": f"matmul-{self.W[0]['name']}",#通过all to all ,将对应的s发送到指定的专家上
                           "left-matrix":['b',self.seq,self.hidden],#理想激活为专家平均处理，即s*topk/ep，极端激活为s*topk(ALl_gather)
                           "right-matrix":[self.W[0]['shape'][0],self.W[0]['shape'][1]], 
                          "output-matrix":['b',self.seq,self.W[0]['shape'][1]],
                          "calc":f"b*{self.seq}*2*{self.hidden}*{self.W[0]['shape'][1]}",
                          "parallel":"tp,epf",
                          "parallel_mode":"col"},True,ioput=self.ioput)
        self.calc_module({"type": "index-aclu-relu",
                           "left-matrix":['b',self.seq,self.W[0]['shape'][1]],
                           "right-matrix":"", 
                          "output-matrix":['b',self.seq,self.W[0]['shape'][1]],
                          "calc":f"b*{self.seq}*{self.W[0]['shape'][1]}",
                          "parallel":"tpf,epf"},True)
        self.calc_module({"type": f"matmul-{self.W[1]['name']}",
                           "left-matrix":['b',self.seq,self.W[0]['shape'][1]],
                           "right-matrix":[self.W[1]['shape'][0],self.W[1]['shape'][1]], 
                          "output-matrix":['b',self.seq,self.W[1]['shape'][1]],
                          "calc":f"b*{self.seq}*2*{self.W[1]['shape'][0]} * {self.W[1]['shape'][1]}",
                          "parallel":"tp,epf",
                          "parallel_mode":"row"},True)#这个输出需要存起来，作为加权的输入
        if self.hidden_dropout > 0.0:
            self.calc_module({"type": "dropout-h",
                           "left-matrix":['b',self.seq,self.W[1]['shape'][1]],
                           "right-matrix":['b',self.seq,self.W[1]['shape'][1]], 
                          "output-matrix":['b',self.seq,self.W[1]['shape'][1]],
                          "calc":f"b*{self.seq}*{self.W[1]['shape'][1]}",#根据RMSNorm公式推导出的近似浮点计算数
                          "parallel":"sp,epf"},True)
        # 与归一化权重进行加权，然后结果乘以--routed-scaling-factor，再加上输入矩阵。
        #每个专家的输出矩阵都是bsh。根据TopK，针对某个确定的token，选定指定的K个专家，在加权时就有（2K-1)个bsh的浮点操作（乘-加乘-加乘...），最后再来一个乘加，总数为2K+1
        if self.flag_moe and args.num_experts is not None and not args.use_fused_moe_token_permute_and_unpermute:#启用的时候，重排和逆重排融合成一个kernel，但第二次通信并不是不存在了，而是以更高效的方式运行，这里直接用取消的激进方式，也是为了抵消一部分其余的被忽略的优化因素
            self.calc_module({"type": f"experts",#加权输出
                           "left-matrix":['b',self.seq,self.hidden],
                           "right-matrix":f"", 
                          "output-matrix":['b',self.seq,self.hidden],
                          "calc":f"(2*Topk+1)*b*{self.seq}*{self.hidden}",
                          "parallel":"tp,ep",
                          "parallel_mode":"row"})#这里单节点内，只存自己专家的输出，通过all2all和所有专家进行数据交换
        self.output_shape = ['b',self.seq,self.hidden]
        #self.activation_list=["input","W1","relu","dropout-h"]

        #是否启用优化
        if optimizations != None:
            self.apply_optimizations(optimizations)
        #if (flag_moe==False and args.add_bias_linear) or (flag_moe==False and args.moe_router_enable_expert_bias):
        if (self.flag_moe==False and args.add_bias_linear):
            self.use_bias=True
        self.calculate_flow()
          
@dataclass
class ffn_swiglu(ModuleConfig):
    """
    ffn_swiglu模块类，继承自ModuleConfig
    """
    def __init__(self,args,flag_moe=False,instance_name: str = "mlp_ffn_swiglu",optimizations:List=None):
        """
        """
        # 调用父类初始化
        super().__init__(args,flag_moe=flag_moe,instance_name=instance_name)
        # 计算输入/输出形状
        self.input_shape = ['b' , self.seq , self.hidden]
         # 定义权重集合 W
        if self.flag_moe and args.num_experts is not None:
            self.W = [{"name": "W1", "shape": ["hidden", "2*h_moe"]},  # Wup = [h, hffn]
                  {"name": "W2", "shape": ["h_moe", "hidden"]},
                  {"name": "WGating_Network", "shape": ["hidden", "route_experts"]}]
        else:
            self.W = [{"name": "W1", "shape": ["hidden", "2*h_ffn"]},  # Wup = [h, hffn]，在实际中，其实是W1=[h,2ffn]
                  {"name": "W2", "shape": ["h_ffn", "hidden"]}] # Wdown = [hffn, h]
        output_acti_flag=True
        if self.flag_moe and args.num_experts is not None:
            output_acti_flag=False
            self.calc_module({"type": f"matmul-{self.W[-1]['name']}",
                           "left-matrix":['b' , self.seq , self.hidden],
                           "right-matrix":[self.W[-1]['shape'][0] , self.W[-1]['shape'][1]], 
                          "output-matrix":['b',self.seq,self.W[-1]['shape'][1]],
                          "calc":f"b*{self.seq}*2*{self.hidden}*{self.W[-1]['shape'][1]}",
                          "parallel":"tp",
                          "parallel_mode":"row"},True,acti=4)#为什么要存4倍大小，profile出来的，我也不知道
            self.calc_module({"type": "mask",#非TopK位置设为负无穷,仅TopK位置保留原分数
                           "left-matrix":['b',self.seq,self.W[-1]['shape'][1]],
                           "right-matrix":['b',self.seq,self.W[-1]['shape'][1]], 
                          "output-matrix":['b',self.seq,self.W[-1]['shape'][1]],#采用索引的方式，和embedding词表一样
                          "calc":f"b*{self.seq}*{self.W[-1]['shape'][1]}",
                          "parallel":""})
            self.calc_module({"type": "softmax-routing",#路由，生成专家选择概率分布
                           "left-matrix":['b',self.seq,self.W[-1]['shape'][1]],
                           "right-matrix":"", 
                          "output-matrix":['b',self.seq,self.W[-1]['shape'][1]],
                          "calc":f"17*b*{self.seq}*{self.W[-1]['shape'][1]}",
                          "parallel":""},True,acti=2)#这里采用32位，因为16位会不稳定
            self.calc_module({"type": "index-Topkchoose",#TopK选择
                           "left-matrix":['b',self.seq,self.W[-1]['shape'][1]],
                           "right-matrix":'', 
                          "output-matrix":['b',self.seq,'Topk'],#采用类似于冒泡索引的方式，和embedding词表一样，这里得到的是Topk的权重
                          "calc":"0",
                          "parallel":""},True)#存了权重
            if args.moe_router_pre_softmax and self.Topk>1:
                self.calc_module({"type": "softmax-topk",#将权重归一化
                           "left-matrix":['b',self.seq,'Topk'],
                           "right-matrix":"", 
                          "output-matrix":['b',self.seq,'Topk'],
                          "calc":f"17*b*{self.seq}*Topk",
                          "parallel":""},True)
            self.calc_module({"type": "softmax-all_expert",#获得全局专家的softmax，这一步主要是为了确保专家之间负载均衡，需要全局视野
                           "left-matrix":['b',self.seq,'route_experts'],
                           "right-matrix":"", 
                          "output-matrix":['b',self.seq,'route_experts'],
                          "calc":f"17*b*{self.seq}*route_experts",
                          "parallel":'ep'},True)
            self.calc_module({"type": "index-comm",#根据权重,用通信分发s
                           "left-matrix":['b',self.seq,'Topk'],
                           "right-matrix":"", 
                          "output-matrix":['b',self.seq,self.hidden],
                          "calc":f"0",
                          "parallel":"ep"},True,moe_flag=True)#每个节点需要保存自己的输入,原输入为[b,s,h]。alltoall之后，每个节点的输入为[1,(1.1)capacity*s*topk*ep/route_experts,h] 
        #f(w1x)开始
        self.calc_module({"type": f"matmul-{self.W[0]['name']}",#Wup
                           "left-matrix":['b' , self.seq , self.hidden],
                           "right-matrix":[self.W[0]['shape'][0] , self.W[0]['shape'][1]], 
                          "output-matrix":['b' , self.seq , self.W[0]['shape'][1]],
                          "calc":f"b*{self.seq}*2*{self.W[0]['shape'][0]} * {self.W[0]['shape'][1]}",
                          "parallel":"tp",
                          "parallel_mode":"col"},True,ioput=self.ioput,moe_flag=flag_moe)#输出存一次，与权重相乘存一次
        #如果不融合，则这个是独立的一步，要激活值，如果融合了，则这一步和下一步算一起，这一步没有激活值。
        #if not args.use_fused_swiglu:
        self.calc_module({"type": "mul-aclu-swish",
                            "left-matrix":['b' , self.seq , f"{self.W[0]['shape'][1]}/2"],
                            "right-matrix":"", 
                            "output-matrix":['b' , self.seq , f"{self.W[0]['shape'][1]}/2"],
                            "calc":f"b*{self.seq}*{self.W[0]['shape'][1]}/2",
                            "parallel":"tpf"},moe_flag=flag_moe)
        #swiglu的激活函数Hadamard
        self.calc_module({"type": "mul-aclu-Hadamard",#在这一步，W1的内部运算才完成，才输出W1x，此时需要存激活
                           "left-matrix":['b' , self.seq , f"{self.W[0]['shape'][1]}/2"],
                           "right-matrix":['b' , self.seq , f"{self.W[0]['shape'][1]}/2"], 
                          "output-matrix":['b' , self.seq , f"{self.W[0]['shape'][1]}/2"],
                          "calc":f"b*{self.seq}*{self.W[0]['shape'][1]}/2",
                          "parallel":"tpf"},True,moe_flag=flag_moe)#激活函数存一次
        #到此，swiglu（f(W1x)）结束
        self.calc_module({"type": f"matmul-{self.W[1]['name']}",
                           "left-matrix":['b' , self.seq , f"{self.W[0]['shape'][1]}/2"],
                           "right-matrix":[self.W[1]['shape'][0] , self.W[1]['shape'][1]], 
                          "output-matrix":['b' , self.seq , self.W[1]['shape'][1]],
                          "calc":f"b*{self.seq}*2*{self.W[1]['shape'][0]} * {self.W[1]['shape'][1]}",
                          "parallel":"tp",
                          "parallel_mode":"row"},output_acti_flag,moe_flag=flag_moe)#权重相乘存一次
        if self.flag_moe and args.num_experts is not None:
            self.calc_module({"type": f"mul-experts",#加权输出
                           "left-matrix":['b',self.seq,self.hidden],
                           "right-matrix":f"", 
                          "output-matrix":['b',self.seq,self.hidden],
                          "calc":f"(2*Topk+1)*b*{self.seq}*{self.hidden}",
                          "parallel":"ep"},True,moe_flag=True)#Topk+1(本家输出)个专家输出，需要N+1次加权，和N次相加
        self.output_shape = ['b',self.seq,self.hidden]            
        #self.activation_list=["input","up","gate","swiglu","Hadamard"]

        #是否启用优化
        if optimizations != None:
            self.apply_optimizations(optimizations)
        #if (flag_moe==False and args.add_bias_linear) or (flag_moe==False and args.moe_router_enable_expert_bias):
        if (self.flag_moe==False and args.add_bias_linear):
            self.use_bias=True
        self.calculate_flow()

@dataclass
class MTP(Self_attention):
    """
    MTP模块类，继承自Self_attention
    """
    def __init__(self,args,instance_name: str = "mtp",optimizations:List=None):
        """
        """
        # 调用父类初始化
        super().__init__(args,instance_name=instance_name)

        # 新增权重 W
                #自身参数
        self.W.append({"name": "Wrms-front", "shape": ["hidden"]})
        self.W.append({"name": "Wrms-input", "shape": ["hidden"]})
        self.W.append({"name": "Wconcat", "shape": ["2*hidden","hidden"]})

        self.calc_module({"type": f"rmsnorm-{self.W[-3]['name']}",
                           "left-matrix":['b' , self.seq , self.hidden],
                           "right-matrix":"", 
                          "output-matrix":['b' , self.seq , self.hidden],
                          "calc":f"5 * b * {self.seq} * {self.hidden}",
                          "parallel":"sp"},True,insert=0)
        self.calc_module({"type": f"rmsnorm-{self.W[-2]['name']}",
                           "left-matrix":['b' , self.seq , self.hidden],
                           "right-matrix":"", 
                          "output-matrix":['b' , self.seq , self.hidden],
                          "calc":f"5 * b * {self.seq} * {self.hidden}",
                          "parallel":"sp"},True,insert=1)
        self.calc_module({"type": f"matmul-{self.W[-1]['name']}",
                           "left-matrix":['b' , self.seq , f"2*{self.hidden}"],
                           "right-matrix":[self.W[-1]['shape'][0],self.W[-1]['shape'][1]], 
                          "output-matrix":['b' , self.seq , self.W[-1]['shape'][1]],
                          "calc":f"b * {self.seq} * 2 * {self.W[-1]['shape'][0]} * {self.W[-1]['shape'][1]}",
                          "parallel":"tp",
                          "parallel_mode":"col"},True,insert=2)                
        
        self.output_shape = ['b',self.seq,self.hidden]           

        #是否启用优化
        if optimizations != None:
            self.apply_optimizations(optimizations)
        if args.add_qkv_bias:
            self.use_bias=True
        self.calculate_flow()

@dataclass
class ModuleCollection():
    """
    模块集合类，支持通过 `集合.方法` 的方式获得实例
    """
    @staticmethod
    def embedding(args,optimizations:List=None):
        instance_name=f"embedding"
        return Embedding(args,instance_name,optimizations)
    @staticmethod
    def output(args,optimizations:List=None):
        instance_name=f"output"
        return Output(args,instance_name,optimizations)
    @staticmethod
    def ffn_swiglu(args,flag_moe,optimizations:List=None):
        instance_name=f"mlp_ffn_swiglu"
        return ffn_swiglu(args,flag_moe,instance_name,optimizations)
    @staticmethod
    def layerNorm(args,optimizations:List=None):
        instance_name=f"layernorm"
        return LayerNorm(args,instance_name,optimizations)
    @staticmethod
    def rms_Norm(args,optimizations:List=None):
        instance_name=f"rms_norm"
        return RMS_Norm(args,instance_name,optimizations)
    @staticmethod
    def ffn_relu(args,flag_moe,optimizations:List=None):
        instance_name=f"mlp_ffn_relu"
        return ffn_relu(args,flag_moe,instance_name,optimizations)
    def self_attention(args,optimizations:List=None):
        instance_name=f"self_attention"
        return Self_attention(args,instance_name,optimizations)  
    def mla(args,optimizations:List=None):
        instance_name=f"mla"
        return MLA(args,instance_name,optimizations) 
    def mtp(args,optimizations:List=None):
        instance_name=f"mtp"
        return MTP(args,instance_name,optimizations) 

@dataclass
class Shared_param_with_embedding(OptimizationConfig):
    instance_name="Shared_param_with_embedding"
    def __init__(self):
        super().__init__(self.instance_name)
        self.W.append({"opera":"share","type": "Wvocab"})
        
@dataclass
class Flash_Attention(OptimizationConfig):
    instance_name="Flash_Attention"
    def __init__(self,mla=False):
        super().__init__(self.instance_name)
        if mla:
            self.activation_list.append({"opera":"add","type": "matmul-Wkvup"})
            self.activation_list.append({"opera":"add","type": "matmul-Wkvup"})
            self.activation_list.append({"opera":"add","type": "matmul-Wkvup"})
        else:
            self.activation_list.append({"opera":"add","type": "matmul-Wqkv"})
        self.activation_list.append({"opera":"del","type": "matmul-score"})
        self.activation_list.append({"opera":"del","type": "softmax"})
        self.activation_list.append({"opera":"del","type": "mask"})
        self.activation_list.append({"opera":"del","type": "dropout-a"})
        self.flow.append({"opera":"del","type": "softmax"})
        self.flow.append({"opera":"del","type": "mask"})
        self.flow.append({"opera":"del","type": "dropout-a"})
        

@dataclass
class Hybrid_MHA_MQA (OptimizationConfig):
    # --group-query-attention \
    # --num-query-groups 16 \
    instance_name="Hybrid_MHA_MQA"
    value_type='int'
    value_range=[]
    #args.num_query_groups=value
    #args.group_query_attention=True
    
    def __init__(self):
        super().__init__(self.instance_name)
        # self.W.append({"opera":"alt","name": "Wk","idx":1, "old_shape": "hidden", "shape": "(hidden*k/a)"})
        # self.W.append({"opera":"alt","name": "Wv","idx":1, "old_shape": "hidden", "shape": "(hidden*k/a)"})
    def set_valuerange(self,num_attention_heads:int):
        self.value_range=[8,num_attention_heads]#最低G=8，再低的话，就不符合实际应用了
        return [self.value_type,self.value_range]    

@dataclass
class DistributedOptimizer(OptimizationConfig):
    # --use-distributed-optimizer \
    instance_name="DistributedOptimizer"
    value_type='bool'
    value_range=[True,False]
    def __init__(self):
        super().__init__(self.instance_name)
        self.parallel_impact.append({"opera":"add","name": "optimizer_state_size"})
    def set_valuerange(self):
        return [self.value_type,self.value_range]
@dataclass
class VirtualPipe(OptimizationConfig):
    # --num-layers-per-virtual-pipeline-stage 1 \
    instance_name="VirtualPipe"
    value_type='int'
    value_range=[]
    num_layers=0
    vppnum=None
    num_layers_per_virtual_pipeline_stage=None
    pp = None
    def __init__(self,num_layers,vppnum=None,num_layers_per_virtual_pipeline_stage=None):
        super().__init__(self.instance_name)
        self.num_layers=num_layers
        self.vppnum=vppnum
        self.num_layers_per_virtual_pipeline_stage=num_layers_per_virtual_pipeline_stage
        up=num_layers//2#必须有pp才能生效，pp最小值为2
        self.value_range=[1,up]
        #print("VirtualPipe init finish")
    def set_valuerange(self):
        return [self.value_type,self.value_range]
    
    def get_vppnum(self,pp):
        if self.vppnum is None or self.pp != pp:
            self.pp = pp
            self.vppnum=int(self.num_layers//(self.pp*self.num_layers_per_virtual_pipeline_stage)) if self.num_layers%(self.pp*self.num_layers_per_virtual_pipeline_stage) == 0 else 1
        return self.vppnum

    def get_constant_vppnum(self):
        return self.vppnum if self.vppnum is not None else 1


@dataclass
class ReCompute(OptimizationConfig):
    # --recompute-granularity selective \
    # --recompute-modules moe
    instance_name="ReCompute"
    value_type='combine'
    value_range=["mlp","moe"]
    recompute_granularity='selective'
    recompute_modules=["mlp","moe"]
    def __init__(self,recompute_granularity='selective',recompute_modules=["mlp","moe"]):
        super().__init__(self.instance_name)
        self.recompute_granularity=recompute_granularity
        self.recompute_modules=recompute_modules
    def set_valuerange(self):
        return [self.value_type,self.value_range]
    # def apply(self, instance):
    #     if instance is ModuleConfig:#针对1,2,3,4参数
    #         instance.recompute_granularity=self.recompute_granularity
    #         instance.recompute_modules=self.recompute_modules
        
@dataclass
class Fused_Cal():
    name="Fused_Cal"

@dataclass
class Op_BasicModule():
    name="Op_BasicModule"
    @staticmethod
    def recompute(recompute_granularity,recompute_modules):
        return ReCompute(recompute_granularity,recompute_modules)

@dataclass
class Op_Embedding():
    name="Op_Embedding"
@dataclass
class Op_Output():
    name="Op_Output"
    @staticmethod
    def share_param():
        return Shared_param_with_embedding()
@dataclass
class Op_LayerNorm():
    name="Op_LayerNorm"
@dataclass
class Op_RMSNorm():
    name="Op_RMSNorm"
@dataclass
class Op_SelfAttention():
    name="Op_SelfAttention"
    @staticmethod
    def flash_attention(mla=False):
        return Flash_Attention(mla)
    @staticmethod
    def hybrid_mha_mqa():
        return Hybrid_MHA_MQA()   
@dataclass
class Op_ffnRelu():
    name="Op_ffnRelu"
@dataclass
class Op_ffnSwiglu():
    name="Op_ffnSwiglu"

@dataclass
class Op_Datapara():
    name="Op_Datapara"
    @staticmethod
    def distributedoptimizer():
        return DistributedOptimizer()

@dataclass
class Op_PipePara():
    name="Op_PipePara"
    @staticmethod
    def virtualpipe(num_layers,num_layers_per_virtual_pipeline_stage):
        return VirtualPipe(num_layers,num_layers_per_virtual_pipeline_stage=num_layers_per_virtual_pipeline_stage)
@dataclass
class Op_ContextPara():
    name="Op_ContextPara"
@dataclass
class Op_UlyssessPara():
    name="Op_UlyssessPara"
@dataclass
class Op_TensorPara():
    name="Op_TensorPara"

@dataclass
class Op_Parallel():
    datapara=Op_Datapara()
    pipepara=Op_PipePara()
    contextpara=Op_ContextPara()
    ulyssesspara=Op_UlyssessPara()
    tensorpara=Op_TensorPara()

@dataclass
class Op_Module():
    basicmodule=Op_BasicModule()
    embedding=Op_Embedding()
    output=Op_Output()
    layernorm=Op_LayerNorm()
    rmsnorm=Op_RMSNorm()
    selfattention=Op_SelfAttention()
    ffnrelu=Op_ffnRelu()
    ffnswiglu=Op_ffnSwiglu()

#优化对权重的改动，对算力的改动作用在flow上，对激活的改动作用在activation_list上
@dataclass
class OptimizerCollection():
    """
    优化集合类，支持通过 `集合.方法` 的方式获得实例
    """
    fused_cal= Fused_Cal()
    module=Op_Module()
    parallel=Op_Parallel()

class NodeMerger:
    def __init__(self):
        self.record_size = []  # {path_hash: node_content}
        self.arch=defaultdict(lambda: defaultdict(list))
        self.acti=defaultdict(lambda: defaultdict(list))
        self.time=defaultdict(lambda: defaultdict(list))
        self.module=defaultdict(lambda: defaultdict(list))
        self.model=defaultdict(lambda: defaultdict(list))
        self.mbs=1
        self.layer=[]
        self.front_flag=True
        self.middle_flag=False
        self.tp_flag=False
        self.cp_flag=False
        self.ep_flag=False
        self.order=[]
    def _generate_node_id(self, node):
        """生成节点唯一标识"""
        serialized = json.dumps({
                'path': node['path'],
                'type': node['type'],
                'size_bytes': node['size_bytes'],
                'children_paths': sorted([child['path'] for child in node.get('children', [])])
            }, sort_keys=True) 
        return hashlib.sha256(serialized.encode()).hexdigest()

    def merge_nodes(self, arch:dict,new_node:dict):
        """合并单个节点"""
        flag=False#默认不合并
        if 'type' not in arch :#第一次创建
            arch['path']=new_node['path']
            arch['type']=new_node['type']
            arch['size_bytes']=new_node['size_bytes']
            arch['children']=new_node['children']
            #合并acti和time
        if 'record_size' not in arch:
            arch['record_size']=[arch['size_bytes']]
        if 'childname' not in arch:
            arch['childname']=[]
            for idx in arch['children']:
                arch['childname'].append(idx['path'])
        if new_node['size_bytes'] not in arch['record_size']:#存在不同
            flag=True
            if 0 == len(new_node['children']):#没有子类，大小又不同，那就是按照大的来
                arch['type']=new_node['type']
            else:    
                for i in range(len(new_node['children'])): 
                    if new_node['children'][i]['path'] not in arch['childname']:
                        arch['children'].append(new_node['children'][i])
                        arch['childname'].append(new_node['children'][i]['path'])
                        arch['size_bytes']=arch['size_bytes']+new_node['children'][i]['size_bytes']
                    else:
                        self.merge_nodes(arch['children'][arch['childname'].index(new_node['children'][i]['path'])],new_node['children'][i])
            arch['record_size'].append(new_node['size_bytes'])
        #到这已经完全相同，接下来合并
        return
     
    def create_module(self,args,arch,module:ModuleConfig):
        pipeline_time=[]
        namelist=arch['path'].split(".")
        module.rename(namelist[-1])
        path_=f'{arch["path"]}._'
        self.acti['name']=self.acti['name'][::-1]
        self.acti['context']=self.acti['context'][::-1]
        ai=len(self.acti['name']) - 1
        while ai >= 0:
            if arch['path'] == self.acti['name'][ai] or path_ in self.acti['name'][ai]:
                self.acti['name'].pop(ai)
                arch['acti']=self.acti['context'].pop(ai)
                if arch['path'] =='embedding.embedding_dropout':
                    self.mbs=arch['acti']['shape'][0][1]
                for j in range(len(arch['acti']['shape'])):
                    module.activation_values_auto.append({"shape": arch['acti']['shape'][j],"num_elements":arch['acti']['num_elements'][j],
                                                "memory_address":arch['acti']['memory_address'][j],"size_bytes":arch['acti']['memory'][j]})
            ai -= 1
            
        self.time['name']=self.time['name'][::-1]
        self.time['context']=self.time['context'][::-1]
        ti =len(self.time['name']) - 1           
        while ti >= 0:
            if arch['path'] == self.time['name'][ti] or path_ in self.time['name'][ti]:
                self.time['name'].pop(ti)
                arch['time']=self.time['context'].pop(ti)  
                if arch['path'] =='embedding.word_embeddings':#这个的时间实际上是流水线并行时，batch进入流水线的时间间隔。              
                    pipeline_time=arch['time']['record'][0]#是[[a]]的形式，固定的，直接取了
                    pipeline_time.pop(0)#第一个是误差，直接删掉就行
                    # try:
                    #     while pipeline_time[1]*100 >pipeline_time[0]:
                    #         pipeline_time.pop(0)
                    # except Exception as e:
                    #     import pdb; pdb.set_trace()
                    # while pipeline_time[1]*100 >pipeline_time[0]:
                    #     pipeline_time.pop(0)
                    num_mbs=(args.global_batch_size//(args.data_parallel_size*args.micro_batch_size))
                    a=np.array(pipeline_time[num_mbs:num_mbs*2])
                    b=np.array(pipeline_time[num_mbs*2:num_mbs*3])
                    c=np.array(pipeline_time[num_mbs*3:num_mbs*4])
                    self.model.pipeline_time_auto=list((a + b + c) / 3)
                else:
                    for j in range(len(arch['time']['shape'])):
                        avg_time=self.choose_data(arch['time']['record'][j])
                        module.time_auto.append({"shape":arch['time']['shape'][j],"time":avg_time})
            ti -=1
        
        for idx in arch['children']:
            if idx['children']== []:
                module.constant_auto=module.constant_auto+idx['size_bytes']      
            elif '_parameters' in idx['path']:              
                for idx_son in idx['children']:
                        if 'device' not in idx_son or idx_son['device'] != 'npu':
                            module.constant_auto=module.constant_auto+idx_son['size_bytes'] 
                        else:
                            module.module_params_auto.append({"name": idx_son['path'].split('.')[-1], "shape": idx_son['shape'],
                                                          "num_elements":idx_son['num_elements'],
                                                          "size_bytes":idx_son['size_bytes']})
            else:
                son_module=ModuleConfig(args=args)
                self.create_module(args,idx,son_module)
                module.constant_auto=module.constant_auto+son_module.constant_auto
                module.children_auto.append(son_module)
        return 
        #import pdb; pdb.set_trace()
        #return self.arch    
        
    def choose_data(self,data:list):
        data= np.array(data)
        Q1=np.percentile(data,25)
        Q3=np.percentile(data,75)
        IQR=Q3-Q1
        lower_bound=Q1-1.5*IQR
        upper_bound=Q3+1.5*IQR

        outliers=data[(data < lower_bound) | (data > upper_bound)]
        normal_data=data[(data >= lower_bound) & (data <= upper_bound)]
        mean_value= np.mean(normal_data)
        return mean_value
    
    def merge_arch(self, arch_list:list,args):
        """合并文件夹内所有JSON文件"""     
        for data in arch_list:
            # 处理根节点
            if 0 == data: continue#防止对应文件缺失的情况，0是初始化值
            if data['content']['size_bytes'] in self.record_size:
                continue
                #print(f"跳过重复节点: {data['path']}")
            else:
                self.merge_nodes(self.arch,data['content'])
                #print(f"新增节点: {data['path']}")
                self.record_size.append(data['content']['size_bytes'])
        #create module
        self.model=ModelConfig(args)
        self.model.init_parallel()   
        for idx in self.arch['children']:
            if idx['children']== []:
                self.model.constant_auto=self.model.constant_auto+idx['size_bytes']
            else:
                module=ModuleConfig(args)
                self.create_module(args=args,arch=idx,module=module)
                self.model.constant_auto=self.model.constant_auto+module.constant_auto
                self.model.module_auto.append(module)

        with open(f"./mm_logs/merge_arch.json", "w") as f:
            json.dump(self.arch, f, indent=2, default=str)
        # with open(f"./mm_logs/merge_model.txt", "w") as f:
        #     json.dump(self.model, f, indent=2, default=str)
        return

    def merge_acti(self, acti_list:list):
        """合并文件夹内所有JSON文件""" 
        self.acti['name']=[]
        self.acti['context']=[]    
        for data in acti_list:
            if 0 == data: continue#防止对应文件缺失的情况，0是初始化值
            data=data['content']
            namelist=list(data.keys())
            for i in range(len(namelist)):
                if namelist[i] not in self.acti['name']:
                    self.acti['name'].append(namelist[i])
                    self.acti['context'].append(data[namelist[i]])
        with open(f"./mm_logs/merge_acti.json", "w") as f:
                json.dump(self.acti, f, indent=2, default=str)
        return
    
    def longest_common_substring(self,a,b):
        if a == []:
            return b
        if b == []:
            return a
        
        m, n= len(a), len(b)
        dp = [[0] * (n+1) for _ in range(m+1)]
        max_len=0
        end_i = 0
        for i in range(1, m+1):
            for j in range(1, n+1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    if dp[i][j] > max_len:
                        max_len = dp[i][j]
                        end_i = i
                else:
                    dp[i][j]=0
        if max_len == 0:
            return a + b
        start_i =end_i - max_len
        common= a[start_i:end_i]
        a_unique=a[:start_i]
        b_unique=b[max_len:]
        return a_unique+common+b_unique
    
    def merge_time(self, time_list:list):
        """合并文件夹内所有JSON文件""" 
        self.time['name']=[]
        self.time['context']=[]    
        #self.time['order']=[]
        for data in time_list:
            if 0 == data: continue#防止对应文件缺失的情况，0是初始化值
            data=data['content']
            namelist=list(data.keys())
            for i in range(len(namelist)):
                if namelist[i] == 'name':
                #     self.time['order']=self.longest_common_substring(self.time['order'],data[namelist[i]])
                    continue
                if namelist[i] not in self.time['name']:
                    self.time['name'].append(namelist[i])
                    self.time['context'].append(data[namelist[i]])
        with open(f"./mm_logs/merge_time.json", "w") as f:
            json.dump(self.time, f, indent=2, default=str)
        self.order=copy.deepcopy(self.time['name'])
        return
    
    def model_automoduleadd(self,namelist:list,parallel:str,module:ModuleConfig=None):
        name=namelist.pop(0)
        if name.startswith("_"):return  
        if module is None:
            module=self.model.get_module_auto(name)
        else:
            module=module.get_children_auto(name)
        if name.isdigit():
            self.front_flag=False
            self.middle_flag=True
            if name not in self.layer:
                self.layer.append(name)
        # if name=="linear_qkv":
        #     import pdb; pdb.set_trace()
        if namelist != []:
            self.model_automoduleadd(namelist,parallel,module)
        else:#先不用shape，因为shape的形状都是根据模型参数定的，模型参数则是一开始固定不能改的。暂时不考虑数据复用的事，每次的数据都是新鲜采集的，因此模型参数一定是相同的
            paralist=parallel.split('_')
            tp=int(paralist[2].split(':')[-1])
            cp=paralist[3].split(':')[-1]
            ep=paralist[4].split(':')[-1]
            sp=eval(f'{tp}*{cp}*{ep}', globals(), {})
            for idx in module.module_params_auto:
                param= idx["num_elements"]*tp
                #if self.tp_flag:有参数的module都会被分割
                param=f'{param}/tp'
                if self.front_flag:
                    self.model.front_param_auto.append(param)
                elif self.middle_flag:
                    self.model.middle_param_auto.append(param)
                else:
                    self.model.back_param_auto.append(param)

            for idx in module.activation_values_auto:
                if idx["memory_address"] in self.model.acti_auto_address: continue
                self.model.acti_auto_address.append(idx["memory_address"])
                acti=(idx["size_bytes"]*sp)/self.mbs
                acti=f'{acti}*b/sp'
                if self.front_flag:
                    self.model.front_acti_auto.append(acti)
                elif self.middle_flag:
                    self.model.middle_acti_auto.append(acti)
                else:
                    self.model.back_acti_auto.append(acti)

            for idx in module.time_auto:
                if len(idx['shape']) < 4 and self.mbs not in idx['shape']:#大于等于4的话，会有注意力头a=32混在里面，32也可能是mbs的大小。大于等于4的话，必定有mbs
                    time=f'{idx["time"]}'
                else:
                    time=idx["time"]/self.mbs
                    time=f'{time}*b'
                if self.front_flag:
                    self.model.front_time_auto.append(time)
                elif self.middle_flag:
                    self.model.middle_time_auto.append(time)
                else:
                    self.model.back_time_auto.append(time)
            self.middle_flag=False
        return
    def model_init(self,parallel,args):
        pp=int(parallel.split('_')[1].split(':')[-1])
        # if self.model.args.reuse_fp32_param:#复用FP32参数副本
        #     self.model.set_optimizer_params("Adam-reuse_fp32_param")
        # else:#optimizer='adam'
        #     self.model.set_optimizer_params("Adam")
        self.model.set_optimizer_params("Adam")
        if self.model.args.add_bias_linear:
            ModuleConfig.use_bias=True
        for name in self.order:
            if name == 'name' or name == '': continue
            self.model_automoduleadd(name.split('.'),parallel)
        self.model.middle_param_auto=self.model.middle_param_auto*pp
        self.model.middle_acti_auto=self.model.middle_acti_auto*pp
        self.model.middle_time_auto=self.model.middle_time_auto*pp   
        if args.untie_embeddings_and_output_weights == False:
            self.model.middle_param_auto.pop()
    
    def auto_parallel(self):
        start_time = time.time()
        solutions=self.model.search_space_create(auto_flag=True)
        time_best,s_best=self.model.costmodel_create(solutions,auto_flag=True)
        print(f"find optimal configuration: {s_best}, find optiaml cost:{time_best},search_cost_time: {time.time() - start_time}")
        self.model.print_bestresult(s_best,auto_flag=True)
        import pdb; pdb.set_trace()
        return

#对于优化选项的填入，两个选择：可以感知细节的，将细节加入框架，在流程中寻优
#不能感知细节的，将优化名加入对应的部分进行存表，直接通过profile得出含该优化和不含该优化的profile时间。
@dataclass
class GPT(ModelConfig):
    """
    Llama2模型配置类，继承自ModelConfig
    包含以下模块结构：
    - front: embedding
    - middle: RMS_Norm, Self_attention, RMS_Norm, ffn_swiglu
    - back: RMS_Norm, output
    使用Adam优化器
    """
    def __init__(self, args,mmlogs_path,search_level=1,print_flag=False):
        """
        初始化Llama2模型配置
        """
        super().__init__(args,mmlogs_path=mmlogs_path,search_level=search_level,print_flag=print_flag)
        
        # 设置优化器为Adam
        #optimizer_selection='fused_adamw'
        # if self.args.reuse_fp32_param:#复用FP32参数副本
        #     self.set_optimizer_params("Adam-reuse_fp32_param")
        # else:#optimizer='adam'
        #     self.set_optimizer_params("Adam")
        listop=[]
        self.set_optimizer_params("Adam")
        #import pdb; pdb.set_trace()
        if print_flag:
            ModuleConfig.print_flag=True
        if self.args.add_bias_linear:
            ModuleConfig.use_bias=True
        if getattr(args, 'use_fused_rmsnorm', False):
            ModuleConfig.ioput=''
        # 设置模型参数
        if self.args.recompute_granularity is not None :
            recompute_modules= getattr(args, 'recompute_modules', ["core_attn"])

            ModuleConfig.apply_basic_optimizations(OptimizerCollection.module.basicmodule.recompute(self.args.recompute_granularity,recompute_modules),print_flag=print_flag)
        
        self.included_modules.front.append(ModuleCollection.embedding(args))
        
        listop.clear()
        k=0
        freq=0
        layer:List[ModuleConfig]
        for i in range(args.num_layers):
            self.flag_moe=False
            layer=[]
            freq=freq+1
            first_k_dense= getattr(args, 'first_k_dense_replace', 0)
            moe_layer_freq=getattr(args, 'moe_layer_freq', 1)
             # --first-k-dense-replace 3 该参数在Megatron里没有实装，故不支持
            if i < first_k_dense:
                self.flag_moe=False
            elif args.num_experts is not None and freq >= moe_layer_freq:#moe-layer-freq
                self.flag_moe=True
                freq=0

            if args.normalization == 'RMSNorm':
                layer.append(ModuleCollection.rms_Norm(args))
            else:
                layer.append(ModuleCollection.layerNorm(args))
            listop.clear()

            self.mla = getattr(args, 'multi_latent_attention', None)
            if self.mla is None: 
                self.mla = getattr(args, 'multi_head_latent_attention', False)
            if self.args.use_flash_attn:
                    listop.append(OptimizerCollection.module.selfattention.flash_attention(self.mla))
            
            if self.mla:
                layer.append(ModuleCollection.mla(args,optimizations=copy.deepcopy(listop)))  
            else:  
                layer.append(ModuleCollection.self_attention(args,optimizations=copy.deepcopy(listop)))
            listop.clear()

            if args.normalization == 'RMSNorm':
                layer.append(ModuleCollection.rms_Norm(args))
            else:
                layer.append(ModuleCollection.layerNorm(args))
            listop.clear()
            
            if self.args.swiglu:
                #import pdb; pdb.set_trace() 
                layer.append(ModuleCollection.ffn_swiglu(args,self.flag_moe))
            # elif: self.args.openai_gelu:
            #     self.included_modules.middle.append(ModuleCollection.ffn_gelu(args,self.flag_moe))
            else:#squared_relu=True
                layer.append(ModuleCollection.ffn_relu(args,self.flag_moe))
            listop.clear()
            k=k+1
            self.included_modules.middle.append(layer)       
#layer  #目前版本不支持mtp参数输入
        # for _ in args.mtp_num_layers:
        #     self.included_modules.back.append(ModuleCollection.mtp(args))
        #     listop.clear()

        if args.normalization == 'RMSNorm':
            self.included_modules.back.append(ModuleCollection.rms_Norm(args))
        else:
            self.included_modules.back.append(ModuleCollection.layerNorm(args))
        listop.clear()

        if False == self.args.untie_embeddings_and_output_weights:
            listop.append(OptimizerCollection.module.output.share_param())    

        self.included_modules.back.append(ModuleCollection.output(args,optimizations=copy.deepcopy(listop)))
        listop.clear()

        self.included_modules_list.append(self.included_modules)
        
        self.init_parallel()
        #生成内存模型初版
        self.memory_model_create()





# ======================== TPDS experiment entry points =======================
def _tpds_candidate_id(record, prefix="cand"):
    payload = json.dumps(_jsonable({"parallel": record.get("parallel"), "strategy": record.get("strategy")}), sort_keys=True)
    return f"{prefix}_{hashlib.sha1(payload.encode()).hexdigest()[:10]}"


def _tpds_make_manifest_record(record, role="measure"):
    out = copy.deepcopy(record)
    out["candidate_id"] = _tpds_candidate_id(out, role)
    out["suite_role"] = role
    return _jsonable(out)


def _tpds_write_manifest(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
    return path



def _tpds_canonicalize_measurement_record(record, num_layers=None):
    """Match measurement candidates to the legality canonicalization used by search."""
    out = copy.deepcopy(record)
    parallel = out.get("parallel") or []
    strategy = out.setdefault("strategy", {})
    if len(parallel) >= 2:
        pp = int(parallel[1])
        vpp = strategy.get("VirtualPipe")
        if vpp not in (None, 0, "None"):
            vpp = int(vpp)
            invalid = pp <= 1
            if num_layers is not None:
                layers = int(num_layers)
                invalid = invalid or layers % (pp * vpp) != 0 or layers <= pp * vpp
            if invalid:
                strategy["VirtualPipe"] = None
    return out


def _tpds_deduplicate_measurement_records(records, num_layers=None):
    unique = {}
    for raw in records:
        record = _tpds_canonicalize_measurement_record(raw, num_layers)
        key = json.dumps(_jsonable({"parallel": record.get("parallel"), "strategy": record.get("strategy", {})}), sort_keys=True)
        old = unique.get(key)
        if old is None or float(record.get("predicted_cost", float("inf"))) < float(old.get("predicted_cost", float("inf"))):
            unique[key] = record
    return list(unique.values())


def apply_candidate_from_env(args):
    """Apply one generated candidate before Megatron initializes process groups.

    Set DTSIR_MEASURE_CANDIDATE_JSON to either the full manifest record or a
    {parallel: [...], strategy: {...}} object. This function must run before
    finish_mpu_init() so TP/PP/CP/EP process groups are created correctly.
    """
    raw = os.getenv("DTSIR_MEASURE_CANDIDATE_JSON", "").strip()
    if not raw:
        return args
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid DTSIR_MEASURE_CANDIDATE_JSON: {e}") from e
    record = _tpds_canonicalize_measurement_record(record, getattr(args, "num_layers", None))
    parallel = record.get("parallel")
    strategy = record.get("strategy", {})
    if not parallel or len(parallel) < 9:
        raise ValueError("Measurement candidate must contain a 9-element 'parallel' list")
    dp, pp, cp, ulyp, tp, sp, ep, mbs, num_mb = parallel[:9]
    args.pipeline_model_parallel_size = int(pp)
    args.tensor_model_parallel_size = int(tp)
    args.context_parallel_size = int(cp) * int(ulyp)
    args.expert_model_parallel_size = int(ep)
    args.micro_batch_size = int(mbs)
    if any(int(v) < 1 for v in parallel[:9]):
        raise ValueError('Parallel degrees and batch counts must be positive; SP disabled is encoded as degree 1, not 0.')
    if int(sp) not in (1, int(tp)):
        raise ValueError('Dense TP sequence-parallel degree must be 1 or TP.')
    args.sequence_parallel = int(sp) > 1
    if int(cp) != 1 or int(ulyp) != 1 or int(ep) != 1:
        raise ValueError('This controlled IR kit supports dense TP/PP/DP only; CP/UP/EP require separate validation.')
    expected_world = int(dp) * int(pp) * int(tp)
    actual_world = int(os.getenv('WORLD_SIZE', expected_world))
    if actual_world != expected_world:
        raise ValueError(f'Candidate world size {expected_world} != launcher {actual_world}')
    if int(args.global_batch_size) != int(dp) * int(mbs) * int(num_mb):
        raise ValueError('Candidate microbatch count disagrees with global batch size')

    vpp = strategy.get("VirtualPipe")
    args.num_layers_per_virtual_pipeline_stage = None if vpp in (None, 0, "None") else int(vpp)
    args.use_distributed_optimizer = bool(strategy.get("DistributedOptimizer", False))

    gqa = strategy.get("Hybrid_MHA_MQA", [False, getattr(args, "num_attention_heads", None)])
    if isinstance(gqa, (list, tuple)) and len(gqa) >= 2:
        args.group_query_attention = bool(gqa[0])
        if gqa[1] is not None:
            args.num_query_groups = int(gqa[1])

    recompute = strategy.get("ReCompute", [None, None])
    if isinstance(recompute, (list, tuple)) and len(recompute) >= 2:
        args.recompute_granularity = recompute[0]
        modules = recompute[1]
        if modules is None:
            args.recompute_modules = None
        elif isinstance(modules, list):
            args.recompute_modules = modules
        elif isinstance(modules, tuple):
            args.recompute_modules = list(modules)
        else:
            args.recompute_modules = [modules]
    os.environ["DTSIR_ACTIVE_CANDIDATE_ID"] = record.get("candidate_id", _tpds_candidate_id(record))
    if getattr(args, "rank", 0) == 0:
        print(f"[DTSIR] measurement candidate={os.environ['DTSIR_ACTIVE_CANDIDATE_ID']} parallel={parallel} strategy={strategy}")
    return args


def _tpds_run_search_once(args, mmlogs_path, search_level, variant, candidate_limit=None,
                          clear_incremental=True, clear_profile=True, keep_records=False):
    runtime = TPDS_RUNTIME
    runtime.set_variant(variant)
    if candidate_limit is not None:
        runtime.config.candidate_limit = int(candidate_limit)
    runtime.reset(clear_incremental=clear_incremental, clear_profile=clear_profile, clear_records=not keep_records)
    t0 = time.perf_counter()
    result = GPT.search_addoptispace_create(copy.deepcopy(args), mmlogs_path=mmlogs_path,
                                            search_level=search_level, cpu_only=True)
    runtime.timings["experiment_total_seconds"] += time.perf_counter() - t0
    best_args, best_conf, best_cost, best_search, device_rank = result
    best_strategy = runtime.strategy_snapshot(best_args) if best_args is not None else None
    payload = runtime.result_dict({
        "best_parallel": best_conf,
        "best_strategy": best_strategy,
        "best_predicted_cost": best_cost,
        "best_search_time": best_search,
        "device_rank": device_rank,
    })
    return result, payload


def _tpds_select_records(records, n, seed=2026):
    records = list(records)
    if n <= 0 or len(records) <= n:
        return records
    rng = random.Random(seed)
    return [records[i] for i in sorted(rng.sample(range(len(records)), n))]


def _tpds_record_key(record, num_layers=None):
    record = _tpds_canonicalize_measurement_record(record, num_layers)
    return json.dumps(_jsonable({
        "parallel": record.get("parallel"),
        "strategy": record.get("strategy", {}),
    }), sort_keys=True)


def _tpds_index_evaluation_records(feasible, rejected, num_layers=None):
    """Index final evaluation outcome for every semantic candidate."""
    indexed = {}
    for raw in rejected:
        record = _tpds_canonicalize_measurement_record(raw, num_layers)
        indexed[_tpds_record_key(record, num_layers)] = record
    for raw in feasible:
        record = _tpds_canonicalize_measurement_record(raw, num_layers)
        key = _tpds_record_key(record, num_layers)
        old = indexed.get(key)
        if old is None or not old.get("predicted_feasible", False):
            indexed[key] = record
        elif float(record.get("predicted_cost", float("inf"))) < float(old.get("predicted_cost", float("inf"))):
            indexed[key] = record
    return indexed


def _tpds_compare_evaluation_records(reference, incremental, rtol=1e-9, atol=1e-9):
    ref_keys, inc_keys = set(reference), set(incremental)
    status_mismatches, cost_mismatches, memory_mismatches = [], [], []
    max_cost_rel_error = 0.0
    max_memory_rel_error = 0.0
    for key in sorted(ref_keys & inc_keys):
        ref, inc = reference[key], incremental[key]
        ref_ok = bool(ref.get("predicted_feasible", False))
        inc_ok = bool(inc.get("predicted_feasible", False))
        if ref_ok != inc_ok:
            status_mismatches.append(key)
            continue
        field = "predicted_cost" if ref_ok else "predicted_peak_memory"
        a, b = ref.get(field), inc.get(field)
        if a is None and b is None:
            continue
        if a is None or b is None:
            (cost_mismatches if ref_ok else memory_mismatches).append(key)
            continue
        a, b = float(a), float(b)
        rel = abs(a-b) / max(abs(a), abs(b), atol)
        if ref_ok:
            max_cost_rel_error = max(max_cost_rel_error, rel)
        else:
            max_memory_rel_error = max(max_memory_rel_error, rel)
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            (cost_mismatches if ref_ok else memory_mismatches).append(key)
    return {
        "reference_candidates": len(ref_keys),
        "incremental_candidates": len(inc_keys),
        "common_candidates": len(ref_keys & inc_keys),
        "missing_from_incremental": len(ref_keys - inc_keys),
        "extra_in_incremental": len(inc_keys - ref_keys),
        "feasibility_mismatches": len(status_mismatches),
        "cost_mismatches": len(cost_mismatches),
        "rejected_peak_memory_mismatches": len(memory_mismatches),
        "max_cost_relative_error": max_cost_rel_error,
        "max_rejected_peak_memory_relative_error": max_memory_rel_error,
        "passed": not (ref_keys ^ inc_keys or status_mismatches or cost_mismatches or memory_mismatches),
    }


def _tpds_strategy_enabled(record, name):
    strategy = record.get("strategy", {})
    parallel = record.get("parallel", [1]*9)
    lname = name.strip().lower()
    if lname == "tp": return int(parallel[4]) > 1
    if lname == "pp": return int(parallel[1]) > 1
    if lname in {"gqa", "hybrid_mha_mqa"}:
        x = strategy.get("Hybrid_MHA_MQA", [False, None])
        return bool(x[0]) if isinstance(x, (list, tuple)) else False
    if lname in {"recompute", "remat"}:
        x = strategy.get("ReCompute", [None, None])
        return bool(x and x[0] is not None)
    if lname in {"vpp", "virtualpipe"}:
        return strategy.get("VirtualPipe") not in (None, 0, "None")
    if lname in {"zero", "distopt", "distributedoptimizer"}:
        return bool(strategy.get("DistributedOptimizer", False))
    return False


def run_tpds_experiment(args, mmlogs_path, search_level=4):
    """Environment-variable driven experiment dispatcher.

    Experiments implemented here operate on the search/evaluation path. Ranking,
    oracle, and compound correctness additionally emit JSONL manifests that can
    be executed by tpds_run_suite.py using the original Megatron training script.
    """
    runtime = TPDS_RUNTIME.refresh(mmlogs_path)
    exp = runtime.config.experiment
    if exp in {"", "off", "none", "measure"}:
        return None
    if getattr(args, "rank", 0) != 0:
        return {"skip": True, "reason": "TPDS experiments run on rank 0 only"}
    os.makedirs(os.path.join(mmlogs_path, "search_data"), exist_ok=True)
    os.makedirs(os.path.join(mmlogs_path, "calc_data"), exist_ok=True)

    print(f"[DTSIR] experiment={exp} variant={runtime.config.variant} incremental={runtime.config.use_incremental} profile_reuse={runtime.config.use_profile_reuse}")

    if exp == "smoke":
        # A deliberately small 8-GPU validation path. Candidate limit is per
        # strategy combination, so keep both dimensions bounded.
        if runtime.config.strategy_limit <= 0:
            runtime.config.strategy_limit = _env_int("DTSIR_SMOKE_STRATEGY_COMBOS", 6)
        if runtime.config.candidate_limit <= 0:
            runtime.config.candidate_limit = _env_int("DTSIR_SMOKE_CANDIDATES_PER_STRATEGY", 12)
        print(f"[DTSIR] smoke strategy_combos={runtime.config.strategy_limit} candidates_per_strategy={runtime.config.candidate_limit} selection={runtime.config.strategy_selection} max_mbs={runtime.config.max_mbs} profile_loops={WARMUP_LOOP_TIME}+{ITERATION_LOOP_TIME}")
        _, base = _tpds_run_search_once(args, mmlogs_path, search_level, "no_inc", clear_profile=True)
        saved_profile = copy.deepcopy(runtime.profile_overlay)
        _, inc = _tpds_run_search_once(args, mmlogs_path, search_level, "full", clear_profile=False)
        runtime.profile_overlay = saved_profile
        base_cost = base.get("best_predicted_cost")
        inc_cost = inc.get("best_predicted_cost")
        rel = None
        if base_cost not in (None, 0, sys.float_info.max) and inc_cost is not None:
            rel = abs(float(inc_cost)-float(base_cost))/abs(float(base_cost))
        same = base.get("best_parallel") == inc.get("best_parallel")
        same_strategy = base.get("best_strategy") == inc.get("best_strategy")
        passed = bool(same and same_strategy and rel is not None and rel <= 1e-9)
        payload = {
            "experiment": "smoke",
            "strategy_combos": runtime.config.strategy_limit,
            "candidates_per_strategy": runtime.config.candidate_limit,
            "strategy_selection": runtime.config.strategy_selection,
            "baseline_no_inc": base,
            "incremental_full": inc,
            "same_best_parallel": same,
            "same_best_strategy": same_strategy,
            "best_cost_relative_error": rel,
            "passed": passed,
        }
        path = runtime.write_result("smoke", payload)
        print(f"[DTSIR] smoke passed={passed} same_best_parallel={same} same_best_strategy={same_strategy} relative_error={rel}")
        print(f"[DTSIR] result: {path}")
        return payload

    if exp in {"breakdown", "ablation", "profile"}:
        if exp == "ablation" and runtime.config.variant == "all":
            rows = []
            for variant in ["full", "no_inc", "no_profile", "naive"]:
                _, one = _tpds_run_search_once(args, mmlogs_path, search_level, variant,
                                                clear_incremental=True, clear_profile=True)
                rows.append(one)
            payload = {"experiment": "ablation", "variants": rows}
        else:
            _, payload = _tpds_run_search_once(args, mmlogs_path, search_level, runtime.config.variant)
        path = runtime.write_result(exp, payload)
        print(f"[DTSIR] result: {path}")
        return payload

    if exp == "equivalence":
        # Baseline and incremental evaluator share exactly the same measured
        # operator-profile evidence; only the recomputation path differs.
        _, base = _tpds_run_search_once(args, mmlogs_path, search_level, "no_inc", clear_profile=True)
        saved_profile = copy.deepcopy(runtime.profile_overlay)
        _, inc = _tpds_run_search_once(args, mmlogs_path, search_level, "full", clear_profile=False)
        runtime.profile_overlay = saved_profile
        base_cost = base.get("best_predicted_cost")
        inc_cost = inc.get("best_predicted_cost")
        rel = None
        if base_cost not in (None, 0, sys.float_info.max) and inc_cost is not None:
            rel = abs(float(inc_cost)-float(base_cost))/abs(float(base_cost))
        same_parallel = base.get("best_parallel") == inc.get("best_parallel")
        same_strategy = base.get("best_strategy") == inc.get("best_strategy")
        payload = {
            "experiment": "equivalence",
            "baseline": base,
            "incremental": inc,
            "same_best_parallel": same_parallel,
            "same_best_strategy": same_strategy,
            "best_cost_relative_error": rel,
            "passed": bool(same_parallel and same_strategy and rel is not None and rel <= 1e-9),
        }
        path = runtime.write_result("equivalence", payload)
        print(f"[DTSIR] result: {path}")
        return payload

    if exp == "scaling":
        sizes = _env_list_int("DTSIR_SCALE_SIZES", [100, 1000, 10000])
        rows = []
        original_limit = runtime.config.candidate_limit
        for size in sizes:
            _, payload = _tpds_run_search_once(args, mmlogs_path, search_level,
                                                runtime.config.variant, candidate_limit=size,
                                                clear_incremental=True, clear_profile=True)
            payload["requested_candidates_per_strategy"] = size
            rows.append(payload)
        runtime.config.candidate_limit = original_limit
        out = {"experiment": "scaling", "variant": runtime.config.variant, "runs": rows}
        path = runtime.write_result("scaling", out)
        print(f"[DTSIR] result: {path}")
        return out

    if exp == "compound":
        # Compare every captured candidate under from-scratch evaluation and
        # dependency-aware incremental evaluation using the same Profile evidence.
        runtime.config.capture_candidates = True
        runtime.config.capture_rejected = True
        num_layers = getattr(args, "num_layers", None)

        _, reference_payload = _tpds_run_search_once(
            args, mmlogs_path, search_level, "no_inc",
            clear_incremental=True, clear_profile=True, keep_records=False)
        reference_feasible = _tpds_deduplicate_measurement_records(
            copy.deepcopy(runtime.candidate_records), num_layers)
        reference_rejected = copy.deepcopy(runtime.rejected_records)
        shared_profile = copy.deepcopy(runtime.profile_overlay)

        runtime.profile_overlay = shared_profile
        _, incremental_payload = _tpds_run_search_once(
            args, mmlogs_path, search_level, "full",
            clear_incremental=True, clear_profile=False, keep_records=False)
        incremental_feasible = _tpds_deduplicate_measurement_records(
            copy.deepcopy(runtime.candidate_records), num_layers)
        incremental_rejected = copy.deepcopy(runtime.rejected_records)

        reference_index = _tpds_index_evaluation_records(
            reference_feasible, reference_rejected, num_layers)
        incremental_index = _tpds_index_evaluation_records(
            incremental_feasible, incremental_rejected, num_layers)
        comparison = _tpds_compare_evaluation_records(reference_index, incremental_index)

        pair_text = os.getenv(
            "DTSIR_COMPOUND_PAIRS",
            "TP+GQA,TP+ReCompute,PP+ReCompute,VPP+ReCompute,GQA+ReCompute,TP+DistributedOptimizer")
        per_pair = _env_int("DTSIR_COMPOUND_N_PER_PAIR", 3)
        seed = runtime.config.sample_seed
        selected_by_key = {}
        for pair_index, pair in enumerate(x.strip() for x in pair_text.split(',') if x.strip()):
            names = [x.strip() for x in pair.split('+')]
            matched = [r for r in incremental_feasible if all(_tpds_strategy_enabled(r, n) for n in names)]
            selected = _tpds_select_records(matched, per_pair, seed + pair_index)
            for record in selected:
                key = _tpds_record_key(record, num_layers)
                item = selected_by_key.get(key)
                if item is None:
                    item = _tpds_make_manifest_record(record, "compound")
                    item["compound_pairs"] = []
                    item["expected_runtime_outcome"] = "success"
                    selected_by_key[key] = item
                item["compound_pairs"].append(pair)

        manifests = list(selected_by_key.values())
        rejected_audit = []
        for key, record in incremental_index.items():
            if not record.get("predicted_feasible", False):
                item = _tpds_make_manifest_record(record, "compound_rejected")
                item["expected_runtime_outcome"] = "not_executed_predicted_oom"
                rejected_audit.append(item)

        manifest_path = os.path.join(runtime.config.log_dir, f"{runtime.config.run_id}_compound_manifest.jsonl")
        rejected_path = os.path.join(runtime.config.log_dir, f"{runtime.config.run_id}_compound_rejected_audit.jsonl")
        _tpds_write_manifest(manifests, manifest_path)
        _tpds_write_manifest(rejected_audit, rejected_path)
        payload = {
            "experiment": "compound",
            "compound_pairs": pair_text,
            "per_pair_requested": per_pair,
            "reference_no_inc": reference_payload,
            "incremental_full": incremental_payload,
            "comparison": comparison,
            "runtime_manifest": manifest_path,
            "runtime_manifest_size": len(manifests),
            "rejected_audit_manifest": rejected_path,
            "rejected_audit_size": len(rejected_audit),
            "passed": bool(comparison["passed"]),
        }
        result_path = runtime.write_result("compound", payload)
        print(f"[DTSIR] compound comparison passed={comparison['passed']} comparison={comparison}")
        print(f"[DTSIR] result: {result_path}")
        print(f"[DTSIR] runtime manifest: {manifest_path}")
        print(f"[DTSIR] rejected audit: {rejected_path}")
        return payload

    if exp in {"ranking", "oracle"}:
        runtime.config.capture_candidates = True
        _, payload = _tpds_run_search_once(args, mmlogs_path, search_level,
                                            runtime.config.variant, keep_records=True)
        feasible = _tpds_deduplicate_measurement_records(copy.deepcopy(runtime.candidate_records), getattr(args, "num_layers", None))
        rejected = copy.deepcopy(runtime.rejected_records)
        seed = runtime.config.sample_seed
        manifests = []
        if exp == "ranking":
            n = _env_int("DTSIR_RANKING_N", 100)
            chosen = _tpds_select_records(feasible, n, seed)
            manifests = [_tpds_make_manifest_record(x, "ranking") for x in chosen]
            # Also save the predicted top-k independently for recall/regret analysis.
            topk = _env_int("DTSIR_RANKING_TOPK", 10)
            payload["predicted_topk"] = sorted(feasible, key=lambda x: x.get("predicted_cost", float('inf')))[:topk]
        elif exp == "oracle":
            n = _env_int("DTSIR_ORACLE_N", 128)
            chosen = _tpds_select_records(feasible, n, seed)
            manifests = [_tpds_make_manifest_record(x, "oracle") for x in chosen]
            payload["oracle_space_size"] = len(manifests)
        manifest_path = os.path.join(runtime.config.log_dir, f"{runtime.config.run_id}_{exp}_manifest.jsonl")
        _tpds_write_manifest(manifests, manifest_path)
        payload["manifest"] = manifest_path
        payload["manifest_size"] = len(manifests)
        result_path = runtime.write_result(exp, payload)
        print(f"[DTSIR] result: {result_path}")
        print(f"[DTSIR] measurement manifest: {manifest_path}")
        return payload

    raise ValueError(f"Unknown DTSIR_EXPERIMENT={exp}")

# ============================================================================

if __name__ == "__main__":
    import shlex
    from types import SimpleNamespace
    def create_training_namespace():
        return argparse.Namespace(num_layers=40, encoder_num_layers=40, decoder_num_layers=None, \
                                  hidden_size=5120, ffn_hidden_size=13824, num_attention_heads=40, \
                                    attention_backend=None, kv_channels=128, \
                                        group_query_attention=False, num_query_groups=1, max_position_embeddings=2048, \
                                            position_embedding_type='rope', relative_attention_num_buckets=32, \
                                                relative_attention_max_distance=128, use_rotary_position_embeddings=False, \
                                                    rotary_base=10000, rotary_percent=1.0, rotary_interleaved=False, \
                                                        rotary_seq_len_interpolation_factor=None, use_rope_scaling=False, \
                                                            rope_scaling_factor=8.0, add_position_embedding=True, mrope_section=None, \
                                                                make_vocab_size_divisible_by=1, normalization='RMSNorm', \
                                                                    norm_epsilon=1e-05, apply_layernorm_1p=False, apply_residual_connection_post_layernorm=False, openai_gelu=False, squared_relu=False, swiglu=True, onnx_safe=None, bert_binary_head=True, untie_embeddings_and_output_weights=True, multi_latent_attention=False, mtp_num_layers=None, mtp_loss_scaling_factor=0.1, attention_dropout=0.0, hidden_dropout=0.0, weight_decay=0.1, start_weight_decay=0.1, end_weight_decay=0.1, weight_decay_incr_style='constant', clip_grad=1.0, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-08, sgd_momentum=0.9, micro_batch_size=4, global_batch_size=128, rampup_batch_size=None, decrease_batch_size_if_needed=False, recompute_granularity=None, check_for_nan_in_loss_and_grad=False, check_for_spiky_loss=False, check_for_large_grads=False, distribute_saved_activations=False, recompute_method=None, recompute_num_layers=None, recompute_modules=None, clone_scatter_output_in_embedding=True, profile=False, profile_step_start=10, profile_step_end=12, iterations_to_skip=[], result_rejected_tracker_filename=None, enable_gloo_process_groups=True, use_pytorch_profiler=False, profile_ranks=[0], record_memory_history=False, memory_snapshot_path='snapshot.pickle', tp_comm_overlap=False, tp_comm_overlap_cfg=None, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_bulk_dgrad=True, tp_comm_bulk_wgrad=True, tp_comm_bootstrap_backend='nccl', use_cpu_initialization=None, empty_unused_memory_level=0, deterministic_mode=False, check_weight_hash_across_dp_replicas_interval=None, calculate_per_token_loss=False, train_sync_interval=None, train_iters=10, train_samples=None, log_interval=1, exit_interval=None, exit_duration_in_mins=None, exit_signal_handler=False, tensorboard_dir=None, masked_softmax_fusion=False, bias_gelu_fusion=False, bias_swiglu_fusion=True, bias_dropout_fusion=True, apply_rope_fusion=True, cross_entropy_loss_fusion=False, cross_entropy_fusion_impl='native', use_flash_attn=True, add_bias_linear=False, add_qkv_bias=False, optimizer='adam', optimizer_cpu_offload=False, optimizer_offload_fraction=1.0, use_torch_optimizer_for_cpu_offload=False, overlap_cpu_optimizer_d2h_h2d=False, pin_cpu_grads=True, pin_cpu_params=True, dataloader_type='single', async_tensor_model_parallel_allreduce=True, no_persist_layer_norm=False, sequence_parallel=True, gradient_accumulation_fusion=False, deprecated_use_mcore_models=False, use_legacy_models=False, manual_gc=False, manual_gc_interval=0, manual_gc_eval=True, tp_comm_split_ag=True, tp_comm_split_rs=True, pipeline_model_parallel_comm_backend=None, seed=1234, data_parallel_random_init=False, init_method_std=0.01, init_method_xavier_uniform=False, lr=1e-06, lr_decay_style='cosine', lr_wsd_decay_style='exponential', lr_decay_iters=None, lr_decay_samples=None, lr_wsd_decay_samples=None, lr_wsd_decay_iters=None, lr_warmup_fraction=0.01, lr_warmup_iters=0, lr_warmup_samples=0, lr_warmup_init=0.0, min_lr=1e-07, override_opt_param_scheduler=False, use_checkpoint_opt_param_scheduler=False, decoupled_lr=None, decoupled_min_lr=None, save=None, save_interval=10, no_save_optim=None, no_save_rng=None, load=None, no_load_optim=True, no_load_rng=True, non_persistent_save_interval=None, non_persistent_ckpt_type=None, non_persistent_global_ckpt_dir=None, non_persistent_local_ckpt_dir=None, non_persistent_local_ckpt_algo='fully_parallel', finetune=False, pretrained_checkpoint=None, ckpt_step=None, perform_initialization=True, use_checkpoint_args=False, use_mp_args_from_checkpoint_args=False, use_tokenizer_model_from_checkpoint_args=True, exit_on_missing_checkpoint=False, use_dist_ckpt_deprecated=False, use_persistent_ckpt_worker=False, auto_detect_ckpt_format=False, dist_ckpt_format_deprecated=None, ckpt_format='torch_dist', ckpt_convert_format=None, ckpt_convert_save=None, ckpt_convert_update_legacy_dist_opt_format=False, ckpt_fully_parallel_save_deprecated=False, ckpt_fully_parallel_save=True, async_save=None, ckpt_fully_parallel_load=False, ckpt_assume_constant_structure=False, dist_ckpt_strictness='assume_ok_unexpected', fp16=True, bf16=False, grad_reduce_in_bf16=False, loss_scale=None, initial_loss_scale=65536.0, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, fp32_residual_connection=False, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=True, accumulate_allreduce_grads_in_fp32=False, fp16_lm_cross_entropy=False, disable_bf16_reduced_precision_matmul=False, tensor_model_parallel_size=8, encoder_tensor_model_parallel_size=0, pipeline_model_parallel_size=1, encoder_pipeline_model_parallel_size=0, pipeline_model_parallel_split_rank=None, decoder_first_pipeline_num_layers=None, decoder_last_pipeline_num_layers=None, num_layers_per_virtual_pipeline_stage=None, num_virtual_stages_per_pipeline_rank=None, microbatch_group_size_per_vp_stage=None, overlap_p2p_comm=False, overlap_p2p_comm_warmup_flush=False, distributed_backend='nccl', distributed_timeout_minutes=10, overlap_grad_reduce=False, defer_embedding_wgrad_compute=False, wgrad_deferral_limit=0, align_grad_reduce=True, ddp_num_buckets=None, ddp_bucket_size=None, ddp_pad_buckets_for_high_nccl_busbw=False, ddp_average_in_collective=False, overlap_param_gather=False, overlap_param_gather_with_optimizer_step=False, align_param_gather=False, scatter_gather_tensors_in_pipeline=True, use_ring_exchange_p2p=False, local_rank=0, lazy_mpu_init=None, account_for_embedding_in_pipeline_split=False, account_for_loss_in_pipeline_split=False, use_distributed_optimizer=False, use_custom_fsdp=False, init_model_with_meta_device=False, data_parallel_sharding_strategy='no_shard', gradient_reduce_div_fusion=True, suggested_communication_unit_size=None, keep_fp8_transpose_cache_when_using_custom_fsdp=False, num_distributed_optimizer_instances=1, use_torch_fsdp2=False, context_parallel_size=1, cp_comm_type=['p2p'], hierarchical_context_parallel_sizes=None, nccl_communicator_config_path=None, use_tp_pp_dp_mapping=False, replication=False, replication_jump=None, replication_factor=2, eval_iters=0, eval_interval=10, test_mode=False, skip_train=False, data_path=['./dataset/enwiki_text_document'], split='10,0,0', train_data_path=None, valid_data_path=None, test_data_path=None, data_args_path=None, per_split_data_args_path=None, data_cache_path=None, mmap_bin_files=True, mock_data=False, seq_length=2048, encoder_seq_length=2048, decoder_seq_length=None, retriever_seq_length=256, sample_rate=1.0, mask_prob=0.15, short_seq_prob=0.1, num_workers=2, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False, create_attention_mask_in_dataloader=False, num_dataset_builder_threads=1, s3_cache_path=None, vocab_size=None, vocab_file=None, merge_file=None, vocab_extra_ids=0, tokenizer_type='Llama2Tokenizer', tokenizer_model='./model_from_hf/llama2-hf/tokenizer.model', tiktoken_pattern=None, tiktoken_num_special_tokens=1000, tiktoken_special_tokens=None, adlr_autoresume=False, adlr_autoresume_interval=1000, ict_head_size=None, biencoder_projection_dim=0, biencoder_shared_query_context_model=False, ict_load=None, bert_load=None, titles_data_path=None, query_in_block_prob=0.1, use_one_sent_docs=False, evidence_data_path=None, retriever_report_topk_accuracies=[], retriever_score_scaling=False, block_data_path=None, embedding_path=None, indexer_batch_size=128, indexer_log_interval=1000, num_classes=1000, img_h=224, img_w=224, num_channels=3, patch_dim=16, classes_fraction=1.0, data_per_class_fraction=1.0, data_sharding=True, head_lr_mult=1.0, vision_pretraining=False, vision_pretraining_type='classify', vision_backbone_type='vit', swin_backbone_type='tiny', mask_type='random', mask_factor=1.0, iter_per_epoch=1250, dino_local_img_size=96, dino_local_crops_number=10, dino_head_hidden_size=2048, dino_bottleneck_size=256, dino_freeze_last_layer=1, dino_norm_last_layer=False, dino_warmup_teacher_temp=0.04, dino_teacher_temp=0.07, dino_warmup_teacher_temp_epochs=30, qk_layernorm=False, expert_model_parallel_size=1, expert_tensor_parallel_size=8, num_experts=None, moe_layer_freq=1, moe_ffn_hidden_size=13824, moe_shared_expert_intermediate_size=None, moe_shared_expert_overlap=False, moe_grouped_gemm=False, moe_use_legacy_grouped_gemm=False, moe_layer_recompute=False, moe_extended_tp=False, moe_use_upcycling=False, moe_router_load_balancing_type='aux_loss', moe_router_dtype=None, moe_router_score_function='softmax', moe_router_topk=2, moe_router_pre_softmax=False, moe_router_num_groups=None, moe_router_group_topk=None, moe_router_topk_scaling_factor=None, moe_router_enable_expert_bias=False, moe_router_bias_update_rate=0.001, moe_aux_loss_coeff=0.0, moe_z_loss_coeff=None, moe_input_jitter_eps=None, moe_per_layer_logging=False, moe_token_dispatcher_type='allgather', moe_enable_deepep=False, moe_permute_fusion=False, moe_expert_capacity_factor=None, moe_pad_expert_input_to_capacity=False, moe_token_drop_policy='probs', q_lora_rank=None, kv_lora_rank=32, qk_head_dim=128, qk_pos_emb_head_dim=64, v_head_dim=128, rotary_scaling_factor=1.0, mscale=1.0, mscale_all_dim=1.0, heterogeneous_layers_config_path=None, heterogeneous_layers_config_encoded_json=None, log_params_norm=False, log_num_zeros_in_grad=False, log_throughput=False, log_progress=False, timing_log_level=0, barrier_with_L1_time=True, timing_log_option='minmax', tensorboard_log_interval=1, tensorboard_queue_size=1000, log_timers_to_tensorboard=False, log_loss_scale_to_tensorboard=True, log_validation_ppl_to_tensorboard=False, log_memory_to_tensorboard=False, log_world_size_to_tensorboard=False, wandb_project='', wandb_exp_name='', wandb_save_dir='', logging_level=None, log_straggler=False, disable_straggler_on_startup=False, straggler_ctrlr_port=65535, straggler_minmax_count=1, run_workload_inspector_server=False, inference_batch_times_seqlen_threshold=-1, max_tokens_to_oom=12000, output_bert_embeddings=False, bert_embedder_type='megatron', flash_decode=False, enable_cuda_graph=False, cuda_graph_warmup_steps=3, external_cuda_graph=False, cuda_graph_scope='full', inference_max_batch_size=8, inference_max_seq_length=2560, inference_dynamic_batching=False, inference_dynamic_batching_buffer_size_gb=40.0, inference_dynamic_batching_buffer_guaranteed_fraction=0.2, inference_dynamic_batching_buffer_overflow_factor=None, inference_dynamic_batching_max_requests_override=None, inference_dynamic_batching_max_tokens_override=None, fp8=None, fp8_recipe='delayed', fp8_margin=0, fp8_interval=1, fp8_amax_history_len=1, fp8_amax_compute_algo='most_recent', fp8_wgrad=True, transformer_impl='transformer_engine', fp8_param_gather=False, first_last_layers_bf16=False, num_layers_at_start_in_bf16=1, num_layers_at_end_in_bf16=1, te_rng_tracker=False, inference_rng_tracker=False, retro_project_dir=None, retro_add_retriever=False, retro_cyclic_train_iters=None, retro_encoder_layers=2, retro_encoder_hidden_dropout=0.1, retro_encoder_attention_dropout=0.1, retro_num_neighbors=2, retro_num_retrieved_chunks=2, retro_attention_gate=1, retro_verify_neighbor_count=True, spec=None, hybrid_attention_ratio=0.0, hybrid_mlp_ratio=0.0, hybrid_override_pattern=None, mamba_state_dim=128, mamba_head_dim=64, mamba_num_groups=8, is_hybrid_model=False, yaml_cfg=None, use_precision_aware_optimizer=False, main_grads_dtype=torch.float32, main_params_dtype=torch.float32, exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32, enable_one_logger=True, one_logger_project='megatron-lm', one_logger_run_name=None, one_logger_async=False, app_tag_run_name=None, app_tag_run_version='0.0.0', enable_ft_package=False, calc_ft_timeouts=False, config_logger_dir='', error_injection_rate=0, error_injection_type='transient_error', rerun_mode='disabled', optimizer_selection='fused_adamw', optimization_level=2, use_fused_rmsnorm=True, use_fused_swiglu=False, context_parallel_algo='megatron_cp_algo', cp_window_size=1, attention_mask_type='causal', use_cp_send_recv_overlap=False, use_fused_ring_attention_update=False, megatron_cp_in_bnsd=False, ulysses_degree_in_cp=None, context_parallel_kv_cache_policy=None, context_parallel_cache_interval=0, use_ulysses_allgather_kv=False, attention_mask_on_cpu=False, adaptive_cp_without_coarse=False, adaptive_cp_dynamic_attn_mask=False, adaptive_cp_only_reschedule=False, adaptive_cp_manually_set_mask_list=False, async_log_allreduce=False, use_fused_rotary_pos_emb=False, use_fused_moe_token_permute_and_unpermute=False, npu_deterministic=False, op_cal_tflops=False, profile_level='level0', profile_with_cpu=False, profile_with_stack=False, profile_with_memory=False, profile_record_shapes=False, profile_save_path='./profile_dir', recompute_activation_function=False, recompute_activation_function_num_layers=None, recompute_norm=False, recompute_norm_num_layers=None, enable_recompute_layers_per_pp_rank=False, unaligned_linear=False, use_ascend_mc2=False, use_ascend_coc=False, coc_mode=-1, coc_parallel_num=1, coc_fused_kernel=False, tp_2d=False, tp_x=1, tp_y=1, enable_overlap_ag_with_matmul=False, enable_overlap_matmul_with_rs=False, enable_backward_overlap_ag_with_matmul=False, recompute_in_bubble=False, recompute_in_advance=False, noop_layers=None, variable_seq_lengths=False, use_multiparameter_pipeline_model_parallel=False, optimize_send_recv_comm=False, pipeline_num_transformer_layers=None, schedules_method=None, dualpipev_dw_detach=False, gemm_gradient_accumulation_fusion=False, moe_tp_extend_ep=False, moe_permutation_async_comm=False, n_shared_experts=None, moe_allgather_overlap_comm=False, moe_alltoall_overlap_comm=False, moe_zero_memory='disable', moe_zero_memory_num_layers=None, moe_fb_overlap=False, moe_unperm2_mem_optim_swap=False, hccl_group_buffer=None, hccl_group_buffer_adaptive=False, hccl_ep_group_buffer_adaptive_factor=-1.0, ema_decay=0.9999, virtual_optimizer=None, dist_train=False, tokenizer_name_or_path=None, tokenizer_not_use_fast=True, param_and_grad_buffer_pad=None, layerzero=False, layerzero_config=None, fsdp2_config_str=None, reuse_fp32_param=False, smart_swap=False, swap_attention=False, swap_modules='input_norm,self_attention,post_attention_norm', compress_dense='disable', disable_gloo_group=False, hccl_slice_size=10485760, swap_optimizer=False, swap_optimizer_times=16, pre_tockens=65536, next_tockens=0, sparse_mode=0, use_fusion_attn_v2=False, square_alibi_mask=False, fill_neg_inf=False, alibi_fusion_attn_type=None, alibi_diagonal_opposite=False, multi_head_latent_attention=False, qk_rope_head_dim=None, qk_nope_head_dim=None, ai_framework='pytorch', rank=0, world_size=8, use_dist_ckpt=True, transformer_pipeline_model_parallel_size=1, data_parallel_size=1, virtual_pipeline_model_parallel_size=None, params_dtype=torch.float16, consumed_train_samples=0, skipped_train_samples=0, consumed_valid_samples=0, reduce_recompute_for_last_chunk=False, padded_vocab_size=32000)  
    def parse_shell_script(script_path):
        """
        解析shell脚本，提取所有参数并转换为Python命名空间
        """
        # 读取shell脚本内容
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取所有变量定义
        variable_pattern = r'^(\w+)=([^#\n]+)'
        variables = {}
        for match in re.finditer(variable_pattern, content, re.MULTILINE):
            var_name, var_value = match.groups()
            # 清理值（去除引号、空格、反斜杠等）
            var_value = var_value.strip().strip('"\'')
            # 处理换行和反斜杠
            var_value = re.sub(r'\\\s*\n\s*', ' ', var_value)
            variables[var_name] = var_value
        
        # 提取所有参数块（DISTRIBUTED_ARGS, GPT_ARGS, DATA_ARGS, OUTPUT_ARGS等）
        param_blocks = {}
        arg_block_pattern = r'(\w+_ARGS)="([^"]*)"'
        
        for match in re.finditer(arg_block_pattern, content, re.DOTALL):
            block_name, block_content = match.groups()
            # 清理块内容（处理换行和反斜杠）
            block_content = re.sub(r'\\\s*\n\s*', ' ', block_content)
            param_blocks[block_name] = block_content
        
        # 提取命令行中的直接参数
        command_line_match = re.search(r'python[^\\]*\s+([^#\n]+)', content)
        command_line_params = ""
        if command_line_match:
            command_line_params = command_line_match.group(1)
            # 替换变量引用
            command_line_params = replace_variables(command_line_params, variables)
        
        # 合并所有参数
        all_params_text = ""
        for block_name in ['DISTRIBUTED_ARGS', 'GPT_ARGS', 'DATA_ARGS', 'OUTPUT_ARGS']:
            if block_name in param_blocks:
                block_content = param_blocks[block_name]
                # 替换变量引用
                block_content = replace_variables(block_content, variables)
                all_params_text += " " + block_content
        
        all_params_text += " " + command_line_params
        
        # 解析参数字符串
        params = parse_parameter_string(all_params_text, variables)
        
        return params
    def replace_variables(text, variables):
        """
        替换文本中的变量引用
        """
        # 处理${VAR}和$VAR形式的变量引用
        def replace_match(match):
            var_name = match.group(1) or match.group(2)
            return variables.get(var_name, match.group(0))
        
        # 匹配${VAR}和$VAR
        pattern = r'\$\{(\w+)\}|\$(\w+)'
        return re.sub(pattern, replace_match, text)
    def parse_parameter_string(param_string, variables):
        """
        解析参数字符串，提取所有--参数
        """
        params = {}
        
        # 使用shlex分割参数（处理引号等特殊情况）
        try:
            tokens = shlex.split(param_string)
        except:
            # 如果shlex分割失败，使用简单空格分割
            tokens = param_string.split()
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith('--'):
                param_name = token[2:]  # 去除--前缀
                python_param_name = param_name.replace('-', '_')
                
                # 检查下一个token是否是值（不是以--开头）
                if i + 1 < len(tokens) and not tokens[i + 1].startswith('--'):
                    param_value = tokens[i + 1]
                    # 处理变量引用
                    param_value = replace_variables(param_value, variables)
                    # 推断类型
                    processed_value = infer_value_type(param_value)
                    params[python_param_name] = processed_value
                    i += 2
                else:
                    # 标志参数（没有值，默认为True）
                    params[python_param_name] = True
                    i += 1
            else:
                i += 1
        
        return params
    def infer_value_type(value):
        """
        推断参数值的Python类型
        """
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        elif value.isdigit():
            return int(value)
        elif re.match(r'^-?\d+\.\d+$', value):
            return float(value)
        elif re.match(r'^\d+\.\d+e[-+]?\d+$', value.lower()):
            return float(value)
        elif ',' in value and not value.startswith('['):
            # 尝试解析逗号分隔的列表（如"10,0,0"）
            try:
                items = [infer_value_type(item.strip()) for item in value.split(',')]
                return items
            except:
                return value
        return value
    def create_python_namespace(params):
        """
        创建Python命名空间对象
        """
        abc = SimpleNamespace()
        
        for param_name, param_value in params.items():
            setattr(abc, param_name, param_value)
        
        return abc
    def generate_python_code(abc_namespace):
        """
        生成Python代码字符串
        """
        code_lines = ["from types import SimpleNamespace", "", "abc = SimpleNamespace()", ""]
        
        # 按参数名排序，使输出更有序
        sorted_params = sorted(vars(abc_namespace).items())
        
        for param_name, param_value in sorted_params:
            if isinstance(param_value, str):
                # 字符串值需要加引号
                value_str = f"'{param_value}'"
            elif isinstance(param_value, list):
                # 列表值
                value_str = str(param_value)
            elif isinstance(param_value, bool):
                # 布尔值
                value_str = str(param_value)
            else:
                value_str = str(param_value)
            
            code_lines.append(f"abc.{param_name} = {value_str}")
        
        return "\n".join(code_lines)
    def parse_training_namespace(script_path):
        params = parse_shell_script(script_path)
        abc = create_python_namespace(params)

        return abc   
    def merge_namespaces(abc, args):
        """
        将abc命名空间对象的值合并到args命名空间对象中
        - 如果args中存在同名属性，则覆盖
        - 如果args中不存在同名属性，则创建新属性
        """
        # 获取abc命名空间的所有属性
        abc_attrs = vars(abc)
        
        # 遍历abc的所有属性
        for attr_name, attr_value in abc_attrs.items():
            # 将值设置到args中（存在则覆盖，不存在则创建）
            setattr(args, attr_name, attr_value)
        
        return args
    
    args= create_training_namespace()
    if len(sys.argv) != 2:
        print("用法: python test_parallel_model.py <shell_script_path>")
        sys.exit(1)
    script_path = sys.argv[1]
    if not os.path.exists(script_path):
        print(f"错误: 文件 {script_path} 不存在")
        sys.exit(1)
    abc = parse_training_namespace(script_path)
    args = merge_namespaces(abc,args)
    args.world_size=args.nproc_per_node * args.nnodes
    args.rank=0
    automm = os.environ.get('AUTOMM')
    data_path=os.getenv('DTSIR_MML_LOGS', '/home/zhangyuhang/users/wjy_hnu/test26/Megatron-LM/mm_logs')
    if automm=='4':
        #if args.rank !=0: time.sleep(360000)
        search_level = os.getenv("SEARCH_LEVEL", 4)  # 默认值为 "1"
        start_time = time.time()
        best_args,best_conf,best_cost_time,best_search_time=GPT.search_addoptispace_create(args,mmlogs_path=data_path,search_level=4,cpu_only=True)
        if args.rank==0:
            model_search=GPT(best_args,mmlogs_path=data_path,search_level=4)
            model_search.search_space_create()
            model_search.print_bestresult(best_conf,print_flag=False)
        args=best_args
        if args.rank==0:
            print(f'dp={args.data_parallel_size}')
            print(f'pp={args.pipeline_model_parallel_size}')
            print(f'cp={args.context_parallel_size}')
            print(f'tp={args.tensor_model_parallel_size}')
            print(f'ep={args.expert_model_parallel_size}')
            print(f'mbs={args.micro_batch_size}')
            print(f'vpp={args.num_layers_per_virtual_pipeline_stage}')
            print(f'distri={args.use_distributed_optimizer}')
            print(f'args.group_query_attention={args.group_query_attention}')
            print(f'args.num_query_groups={args.num_query_groups}')
            print(f'args.recompute_granularity={args.recompute_granularity}')
            print(f'args.recompute_modules={args.recompute_modules}')
            print(f'args.num_layers_per_virtual_pipeline_stage:{args.num_layers_per_virtual_pipeline_stage}')
            import shutil
            if os.path.exists(f'{data_path}/search_data'):
                # 递归删除整个文件夹
                shutil.rmtree(f'{data_path}/search_data')
            # 重新创建一个空的同名文件夹
            os.makedirs(f'{data_path}/search_data')
        #import pdb; pdb.set_trace()
    elif automm=='1':
        if args.rank !=0: time.sleep(360000)
        search_level = os.getenv("SEARCH_LEVEL", 4)  # 默认值为 "1"
        start_time = time.time()
        model=GPT(args,mmlogs_path=data_path,search_level=4,print_flag=True)
        solutions=model.search_space_create()
        # if len(solutions) == 0:
        #     print('solution find num is None!!!')
        #     import pdb; pdb.set_trace()
        time_best,s_best=model.costmodel_create(solutions)
        print(f"find optimal configuration: {s_best}, find optiaml cost:{time_best},search_cost_time: {time.time() - start_time}")
        #s_best=[2, 1, 4, 1,1, 1, 1, 1, 64]
        model.print_bestresult(s_best)
        import pdb; pdb.set_trace() 
        args.data_parallel_size=s_best[0]
        args.pipeline_model_parallel_size=s_best[1]
        #args.transformer_pipeline_model_parallel_size
        args.context_parallel_size=s_best[2]*s_best[3]
        args.tensor_model_parallel_size=s_best[4]
        args.expert_model_parallel_size=s_best[6]
        #args.expert_tensor_parallel_size
        args.micro_batch_size=s_best[7]
        print(f'dp={args.data_parallel_size}')
        print(f'pp={args.pipeline_model_parallel_size}')
        print(f'cp={args.context_parallel_size}')
        print(f'tp={args.tensor_model_parallel_size}')
        print(f'ep={args.expert_model_parallel_size}')
        print(f'mbs={args.micro_batch_size}')


    # # Temporary for transition to core datasets
    # train_valid_test_datasets_provider.is_distributed = True
    
    # pretrain(
    #     train_valid_test_datasets_provider,
    #     model_provider,
    #     ModelType.encoder_or_decoder,
    #     forward_step,
    #     args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    # )


























#重计算：开了就不存激活，然后反向计算时间=反向+正向。checkpoint和recompute重复了，应该是checkpoint作废了。类似batch作废一样
#recompute_activations=False, recompute_granularity=None
#distribute_saved_activations=False, recompute_method=None, recompute_num_layers=None
#checkpoint_activations=False


#tp通讯重叠选项，在costmodel里将重叠率再细分一下即可。ag=allgather，rs=reduce_scatter，rs_dgrad=rs和梯度计算异步重叠，bulk_dgrad=批量梯度通信重叠，wgrad=权重梯度,dgrad=数据梯度
#tp_comm_overlap=False, tp_comm_overlap_cfg=None, tp_comm_overlap_ag=True, tp_comm_overlap_rs=True, tp_comm_overlap_rs_dgrad=False, tp_comm_bulk_dgrad=True, tp_comm_bulk_wgrad=True
#tp_comm_split_ag=True, tp_comm_split_rs=True

#融合算子的启用选项，bias的融合算子==不带偏置的对应算子， apply_rope_fusion=融合旋转位置编码（RoPE）的计算步骤。将正弦/余弦位置映射与向量旋转操作合并

#use_fused_rotary_pos_emb=False,算子内核层，替换标准RoPE计算为融合CUDA内核
# apply_rope_fusion=True,图优化层，将RoPE与Attention/QKV计算合并

#rope算子目前没做，主要是因为找不到具体的融合步骤，其次是其作为必选项存在于每个模型，比较时可以忽略。
# 三是因为其不涉及通信和权重的并行影响（无权重且不需要整体计算）（位置信息在序列并行时也已经记录，可以单独进行编码）。

#masked_softmax_fusion=False, bias_gelu_fusion=True, bias_swiglu_fusion=True, bias_dropout_fusion=True,  cross_entropy_loss_fusion=False，add_bias_linear=False(与bias融合算子协同优化),

#将all_reduce拆成ag和rs.delay_grad_reduce=延迟梯度聚合时机，等待反向计算完成后再启动AllReduce.与async_tensor_model_parallel_allreduce=True联用，避免通信阻塞计算流
#(这些都不管，都属于通信的通用优化，默认开启，不做寻优---启发式选择)
#async_tensor_model_parallel_allreduce=True,delay_grad_reduce=True。先是异步ar，然后等待ar全部完成后再grad

#scatter_gather_tensors_in_pipeline=True（在流水线切分前将完整张量分散到设备组）, use_ring_exchange_p2p=False（p2p通信中禁用环形通信（Ring-Exchange），默认使用更高效的树状广播（Tree Broadcast）
#vocab_size=None表示无需手动设置，避免与分词器实际词表冲突
#overlap_p2p_comm=True(p2p->pipeline to pipeline)#默认开启
#optimize_send_recv_comm=False, 发送/接收通信的重叠优化（如与计算并行执行）#如果是p2p场景，则又重复了
#optimize_vpp_send_recv_comm=False 

#num_workers=2,控制数据加载时并行子进程数量，影响数据预处理和I/O效率
# reset_attention_mask=False,
# ​作用​：控制是否对拼接文档（Sample Packing）生成块对角掩码（Block Diagonal Mask）。
#​False效果​：使用标准因果掩码（Causal Mask），允许不同文档间注意力交互，简化计算但可能引入跨文档干扰
#​True效果​：生成隔离文档的块对角掩码，避免干扰但增加计算开销（如32K序列掩码需1GB显存）

#create_attention_mask_in_dataloader=True,（预生成）在数据加载阶段生成注意力掩码，而非模型计算时动态生成。
# biencoder_shared_query_context_model=False。双编码器（bi-encoder）结构中，查询（Query）和上下文（Context）是否共享权重。（检索增强生成（RAG）功能才有可能用，一般不管）

# moe_token_drop_policy='probs',按概率阈值丢弃

#moe_pad_expert_input_to_capacity=False, 是否将输入填充至固定容量。False表示动态适应实际token数。
# moe_layer_recompute=False.是否对MoE层激活值重计算。False表示保存全部中间结果。 
#moe_adaptive_recompute_activation=False, moe_adaptive_recompute_activation_scale=2.0,MOE层的自适应重计算，通过scale控制自适应度

#enable_token_rearrange_opt=False。#Token重排优化（如Ring-Exchange）。跨节点通信时，AllGather带宽消耗增加30%-50%，尤其影响expert_model_parallel_size>1场景
#use_rts=False 使用Reduce-Scatter替代AllGather
#moe_permutation_async_comm=False,异步通信调度token置换。GPU等待通信同步，利用率降至70%以下（对比异步模式>90%）
#moe_no_drop=False,。是否允许丢弃低概率token
#moe_dynamic_padding=False, 禁用专家输入的动态填充。（​动态填充​：按需分配显存，但增加碎片整理开销）
#moe_tp_extend_ep=False,  是否将张量并行（TP）扩展至专家并行（EP）组内。即在EP可以生效的地方，将TP转化为EP生效
#use_fused_moe_token_permute_and_unpermute=False, 融合Kernel处理token置换，未融合时需2次显存读写（Permute + Unpermute），融合后单次完成，速度提升20%。

#gemm_gradient_accumulation_fusion=False,GEMM（矩阵乘）操作的梯度累积融合优化。启用条件​：需安装CUDA扩展（--cpp_ext --cuda_ext）且CUDA≥11；

#moe_alltoall_overlap_comm=False, moe_allgather_overlap_comm=False, #AlltoAll/AllGather通信与专家计算的重叠

#moe_experts_pipeline_degree=0, #deepseek:专家级流水线并行（专家组内无流水线切分）,没听过，不管
# moe_zero_memory='disable', #禁用MoE层的Zero内存优化（专家参数全驻留单卡）'disable''stage1''stage2' ,moe_zero_memory_num_layers=None

#moe_bmm_mc2=False, #专家计算的批矩阵乘融合（BatchMatMul+ReLU）
#cp_window_size=1, #上下文并行滑动窗口大小为1（无重叠切分）。（语言建模任务增至2-4，提升局部注意力覆盖率（如cp_window_size=2））
#use_cp_send_recv_overlap=False,上下文并行（Context Parallelism, CP）中通信与计算的重叠优化 
#use_fused_ring_attention_update=False, 环形注意力（Ring Attention）的融合Kernel更新

# rope_scaling_type=None。RoPE的长度外推（如NTK或Linear Scaling）

#use_ulysses_allgather_kv=False, Ulysses算法优化KV矩阵的AllGather操作
#标准AllGather	O(N)次/Ulysses	O(logN)次（减少50%）
#启用条件​：序列长度 ≥ 8K 且GPU数 ≥ 64（8*8的拓扑才行）

#recompute_module_list=None, recompute_type=2,recompute_type=1：仅重计算注意力模块（显存节省20%）recompute_type=2：重计算所有模块（显存节省40%，计算开销增加25%）
#recompute_module_list = ["attention", "mlp"]  # 仅重计算高显存模块
#context_parallel_cache_interval=0。#memory_fragmentation=False.显存碎片整理.长序列训练（>32K）时碎片导致OOM风险提升30%。启用 True 并配合 context_parallel_cache_interval=10（每10步整理一次）

#automated_pipeline=False, automated_pipeline_perf=False, 自动流水线分析和划分，一起开启==启用自动流水线优化
#（推理）pre_tockens=65536,prefill阶段预输入


# use_pipe_experts=False,第一次A2A时的通信隐藏。

# variable_seq_lengths=False, 

# adaptive_recompute_device_size=-1, #禁止自适应重计算
# adaptive_recompute_profiling_step=10, 每10步分析重计算收益
# adaptive_recompute_device_swap=False, 禁用显存-内存交换
# enable_recompute_layers_per_pp_rank=False,全局统一重计算策略

#recompute_activation_function=False, recompute_activation_function_num_layers=None, #激活函数的重计算，显存瓶颈时设为 True，配合 recompute_activation_function_num_layers=12（仅对前12层启用）
# recompute_norm=False, recompute_norm_num_layers=None（生效层数）, 归一化层的重计算。百亿级模型（如LLaMA-70B）可对首尾层启用（recompute_norm_num_layers=2）
# recompute_in_bubble=False, 禁用流水线气泡期的重计算。
# recompute_in_advance=False,禁用预重计算（提前计算激活值）
#swap_attention=False,禁用Attention矩阵的显存-内存交换
#swap_modules='input_norm,self_attention,post_attention_norm', 
# adaptive_memory_optimization=False, 启用后系统自动识别高频访问数据，将关键参数（如Attention权重）保留在GPU显存，其余数据交换到内存，兼顾效率与显存
# use_fusion_attn_v2=False,#华为特供
#use_multiparameter_pipeline_model_parallel=False,
#ampipe_degree=1, 
#ampipe_tp_sp_comm_overlap=False,

#
