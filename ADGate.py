# Define the basic structure of ADGates
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from dataclasses import dataclass, field

from .decomposition import *
from .tensor_ops import *
from .gates import *
from .einsum_tools import *
from .build import *

from typing import List, Tuple, Optional, Union, Dict, Any, Callable

#不要写的过于复杂了，为了可维护性，只保留一个Gatespec dataclass，一个ADgate的nn.Module类就可以了

@dataclass
class GateSpec:
    name: str
    gate_type: str
    qubit_dims: List[int]
    position: Optional[List[int]] = None
    control: Optional[List[int]] = None
    dtype: mx.Dtype = mx.complex64
    params_shape: Optional[Dict[str, Tuple[int, ...]]] = None
    params_init: Union[str, Dict[str, str]] = "normal"
    other_params: Optional[Dict[str, Any]] = None


class ADgate(nn.Module):
    def __init__(self, spec: GateSpec):
        super().__init__()
        self.spec = spec
        self.name = spec.name.lower()
        self.tensor: Optional[mx.array] = None

        self.params: Dict[str, mx.array] = {}
        if spec.gate_type == 'parameter':
            if not spec.params_shape:
                raise ValueError("params_shape must be provided for parameter gates")
            
            for key, shape in spec.params_shape.items():
                init_way = spec.params_init[key] if isinstance(spec.params_init, dict) else spec.params_init
                v = self._init_param(shape, init_way)
                self.params[key] = v
                self.register_parameter(key, v) #注册参数(在mlx应该用不到)

        self.renew_gate() #生成相应的矩阵形式

    def _init_param(self, shape: List[int], init_way: str) -> mx.array:
        if init_way == 'zeros':
            return mx.zeros(shape, dtype = self.spec.dtype)
        elif init_way == 'ones':
            return mx.ones(shape, dtype = self.spec.dtype)
        elif init_way == 'uniform':
            return mx.random.uniform(shape, dtype = self.spec.dtype)
        elif init_way == 'normal':
            return mx.random.normal(shape, dtype = self.spec.dtype)
        else:
            raise ValueError("init_way must be one of 'zeros', 'ones', 'uniform', 'normal'")
        
    def _get(self, key: str) -> mx.array:
        if key in self.params:
            return self.params[key]
        elif key in self.spec.other_params:
            return self.spec.other_params[key]
        else:
            raise ValueError(f"{key} not found in params or other_spec.")
    
    #用注册表来获得门的矩阵形式
    BUILD_GATE: Dict[str, Callable[["ADgate"], mx.array]] = {}

    @classmethod
    def register_gate(cls, *names: str):
        def wrapper(fn):
            for n in names:
                cls.BUILD_GATE[n] = fn
            return fn
        return wrapper


    def postprocess(self, Uloc: mx.array):
        U = Uloc.astype(self.spec.dtype)
        if U.ndim != 2 or U.shape[0] != U.shape[1]:
            raise ValueError(f"Gate {self.name} must be square, got {U.shape}.")
        return U

    def renew_gate(self):
        if self.name not in self.BUILD_GATE:
            raise NotImplementedError(f"Unknown gate: {self.name}")

        Uloc = self.BUILDERS[self.name](self)
        self.tensor = self.postprocess(Uloc)
        return self.tensor

    def forward(self):
        return self.tensor

    @staticmethod
    def latent2unitary(g: mx.array):
        u, _, v = mx.linalg.svd(g)
        return u @ v