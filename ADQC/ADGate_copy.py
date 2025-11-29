# Define the basic structure of ADQC
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from dataclasses import dataclass, field

from .decomposition import *
from .tensor_ops import *
from .gates import *
from .einsum_tools import *

from typing import List, Tuple, Optional, Union, Dict, Any, Callable

#利用dataclass严格定义参数的相关结构

#参数类，标注参数门的结构
@dataclass
class ParaSpec:
    shape: Tuple[int, ...] #参数门的形状
    init: str = "normal" #随机生成参数的方式，normal为正态分布，uniform为均匀分布等等
    dtype: mx.Dtype = mx.float32 #参数门的数据类型
    transform: Optional[str] = None #对参数进行变换的形式，如对参数进行正交化等

#门类，标注门的结构
@dataclass
class GateSpec: 
    name: str #门的名称
    gate_type: str #门的类型：参数门或者非参数门
    qubit_dims: List[int] #量子比特数量
    position: Optional[List[int]] = None #门的作用位置
    control: Optional[List[int]] = None  #门的控制位置
    dtype: mx.Dtype = mx.complex64 #门中参数的数据类型

    para_spec: Optional[Dict[str, ParaSpec]] = None #如果是参数门，门的可变参数

    other_spec: Optional[Dict[str, Any]] = None #如果是非参数门，门的不可变参数

    requires_grad: bool = True #是否需要梯度
    no_grad: bool = False #是否无梯度

    bulid_key: Optional[str] = None #门的构建关键字，用于构建门的函数

class ParamState:
    #参数状态类，用于保存参数的状态，包括参数的值和是否可训练
    def __init__(self, spec: GateSpec):
        self._values: Dict[str, mx.array] = {} #参数及参数值
        self._trainable: Dict[str, bool] = {} #参数名及是否可训练
        for key, ps in spec.para_spec.items():
            self._values[key] = self._init_param(ps)
            self._trainable[key] = spec.requires_grad

    def _init_param(self, ps: ParaSpec) -> mx.array:
        if ps.init == 'zeros':
            v = mx.zeros(ps.shape, dtype = ps.dtype)
        elif ps.init == 'ones':
            v = mx.ones(ps.shape, dtype = ps.dtype)
        elif ps.init == 'uniform':
            v = mx.random.uniform(ps.shape, dtype = ps.dtype)
        else:
            v = mx.random.normal(ps.shape, dtype = ps.dtype)
        return v
    
    def get(self, key: str) -> mx.array:
        return self._values[key]
    
    def set(self, key: str, value: mx.array) -> None:
        self._values[key] = value

    def items(self):
        return self._values.items()
    
    def trainable(self) -> List[str]:
        return [k for k, v in self._trainable.items() if v]
    
    def freeze(self, key: Optional[List[str]] = None) -> None:
        if key is None:
            for k in self._trainable:
                self._trainable[k] = False
        else:
            for k in key:
                self._trainable[k] = False
    
    def unfreeze(self, key: Optional[List[str]] = None) -> None:
        if key is None:
            for k in self._trainable:
                self._trainable[k] = True
        else:
            for k in key:
                self._trainable[k] = True

def _get_arg(spec: GateSpec, state: Optional[ParamState], key: str):
    if spec.para_spec and key in spec.para_spec:
        if state is None:
            raise ValueError(f"ParamState required for param '{key}' in gate {spec.name}.")
        return state.get(key)

    if not spec.other_spec or key not in spec.other_spec:
        raise KeyError(f"Missing other_spec['{key}'] for gate {spec.name}.")
    return spec.other_spec[key]

def postprocess(Uloc: mx.array, spec: GateSpec, is_fixed: bool = False) -> mx.array:
    U = Uloc.astype(spec.dtype)

    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f"Gate {spec.name} must be a square matrix, got shape {U.shape}.")

    return U


def bulid_gate(spec: GateSpec, state: Optional[ParamState] = None) -> mx.array:
    if spec.name == "spin_rorate":
        j = spec.other_spec["j"]
        n = spec.other_spec["n"]
        theta = _get_arg(spec, state, "theta")

        Uloc = spin_rotate(j, theta, n)
        return postprocess(Uloc, spec, is_fixed=(spec.gate_type != 'param'))
