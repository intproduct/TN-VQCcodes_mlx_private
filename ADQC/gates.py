#一些常用的量子门，包括hadamard、pauli、rorate、phase_shift、spin_rotate
#以及SU(2)生成元及旋转矩阵（即spin算符），二体交换门，通用delta张量

import mlx.core as mx
import numpy as np
import math
from typing import List, Tuple, Optional, Union, Dict

def hadamard() -> mx.array:
    return mx.array([[1.0, 1.0], [1.0,-1.0]], dtype = mx.complex64) / math.sqrt(2)

def cnot() -> mx.array:
    return mx.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]], dtype = mx.complex64)

def cz() -> mx.array:
    return mx.diag(mx.array([1.0, 1.0, 1.0, -1.0], dtype = mx.complex64))

def s_gate() -> mx.array:
    return mx.array([[1.0, 0.0], [0.0, 1j]], dtype = mx.complex64)

def t_gate() -> mx.array:
    return mx.array([[1.0, 0.0], [0.0, mx.exp(1j * math.pi / 4)]], dtype = mx.complex64)
                   
def crz(theta: mx.array) -> mx.array:
    return mx.diag(mx.array(
        [1.0, 1.0,
        mx.exp(-1j * theta / 2),
        mx.exp(1j * theta / 2)]
    , dtype = mx.complex64))

def pauli_ops(name: str) -> mx.array:
    if name == 'x' or name == 'pauli_x':
        return mx.array([[0.0, 1.0],[1.0, 0.0]], dtype = mx.complex64)
    elif name == 'y' or name == 'pauli_y':
        return mx.array([[0.0, -1.0j],[1.0j, 0.0]], dtype = mx.complex64)
    elif name == 'z' or name == 'pauli_z':
        return mx.array([[1.0, 0.0],[0.0, -1.0]], dtype = mx.complex64)
    elif name == 'id':
        return mx.array([[1.0, 0.0],[0.0, 1.0]], dtype = mx.complex64)
    
def rotate_ops(para: Optional[dict], way:'str' = 'parameters') -> mx.array:
    if way == 'parameters':
        if para == None:
            para = {
                'alpha': mx.random.uniform([]),
                'beta': mx.random.uniform([]),
                'gamma': mx.random.uniform([]),
                'theta': mx.random.uniform([])
            }
        alpha, beta, gamma, theta = para['alpha'], para['beta'], para['gamma'], para['theta']
        gate = mx.ones([2,2], dtype = mx.complex64)
        gate[0,0] = mx.exp(1j * (gamma - alpha / 2 - beta / 2)) * mx.cos(theta / 2)
        gate[0, 1] = -mx.exp(1j * (gamma - alpha / 2 + beta / 2)) * mx.sin(theta / 2)
        gate[1, 0] = mx.exp(1j * (gamma + alpha / 2 - beta / 2)) * mx.sin(theta / 2)
        gate[1, 1] = mx.exp(1j * (gamma + alpha / 2 + beta / 2)) * mx.cos(theta / 2)
        return gate.astype(mx.complex64)
    elif way == 'angles':
        if para == None:
            para = {
                'theta': 2 * math.pi * mx.random.uniform([]),
                'direction': 'x'
            }
        theta, direction = para['theta'], para['direction']
        op = pauli_ops(direction)
        return mx.matrix_exp(-1j * theta / 2 * op).astype(mx.complex64)
    else:
        raise ValueError('way must be parameters or angles')    


def phase_shift(theta: mx.array) -> mx.array:
    return mx.array([[1.0, 0.0],[0.0, mx.exp(1j * theta)]], dtype = mx.complex64)

def spin_ops(j:float) -> Dict:
    #直接通过SU(2)的矩阵表示，j为自旋量子数，返回Jx, Jy, Jz
    dims = 2 * j + 1
    value_dim = mx.arange(j, -j-1, -1)
    Jp = mx.zeros([dims, dims])
    Jm = mx.zeros([dims, dims])
    Jz = mx.diag(value_dim)
    for i,m in enumerate(value_dim[:-1]):
        Jp[i, i+1] = mx.sqrt((j - m) * (j + m + 1))
    Jm = Jp.conj().T
    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)
    return {'Jx': Jx, 'Jy': Jy, 'Jz': Jz}

def spin_rotate(j: float, theta: float, n: mx.array) -> mx.array:
    Jx, Jy, Jz = spin_ops(j)
    n = n / mx.linalg.norm(n)
    Jn = n[0] * Jx + n[1] * Jy + n[2] * Jz
    return mx.matrix_exp(-1j * theta * Jn)

def binary_string(num: int) -> List[str]:
    assert num>=0
    if num == 0:
        return ['0']
    else:
        length = len(bin(num-1)[2:]) #bin获得前缀为'0b'的二进制字符串，[2:]去除前缀
        return [format(n, f'0{length}b') for n in range(num)] #format将数字n转为二进制字符串，0表示位数不足length则前补0，b表示二进制，不显示

def swap() -> mx.array:
    return mx.eye(4).reshape(2,2,2,2).transpose(1,0,2,3) #二体交换门生成

def super_diag_tensor(dim: int, order: int, dtype: Optional[mx.Dtype]) -> mx.array:
    shape = [dim] * order
    delta = mx.zeros(shape, dtype)
    idx = mx.arange(dim)
    delta[tuple([idx]*order)] = 1.0
    return delta  #通用delta张量

def qudit_cnot(dim: int) -> mx.array:
    #对应高维控制操作
    gate = mx.eye(dim * dim, dtype=mx.complex64)
    control_value = dim - 1
    for j in range(dim):
        input_idx = control_value * dim + j
        output_idx = control_value * dim + (j + 1) % dim
        gate = gate.at[input_idx, input_idx].set(0)
        gate = gate.at[input_idx, output_idx].set(1)
    return gate

def toffoli() -> mx.array:
    gate = mx.eye(8, dtype = mx.complex64)
    gate = gate.at[6,6].set(0)
    gate = gate.at[6,7].set(1)
    gate = gate.at[7,6].set(1)
    gate = gate.at[7,7].set(0)
    return gate

#最关键的隐门没有写
def latent_gate(pos: Optional[List[int]], dims: Optional[List[int]],
                init_way: Optional[str], dtype: Optional[mx.Dtype]) -> mx.array:
    if pos == None:
        ndim = 2
    else:
        ndim = len(pos)
    if dims is None:
        dims = [2] * ndim
    dim_t = math.prod(dims)
    if init_way == 'identity':
        gate = mx.eye(dim_t) + 1e-5 * mx.random.normal(shape=(dim_t, dim_t))
    else:
        gate = mx.random.normal(shape=(dim_t, dim_t))
    return gate.astype(dtype)

#任意门（我不记得有没有实现了，但是先写出来吧）
def arbitrary_gate(shape: List[int], form: str) -> mx.array:
    if form == 'zeros':
        return mx.zeros(shape,)
    elif form == 'ones':
        return mx.ones(shape)
    elif form == 'uniform':
        return mx.random.uniform(shape)
    elif form == 'normal':
        return mx.random.normal(shape)
    else:
        raise ValueError("init_way must be one of 'zeros', 'ones', 'uniform', 'normal'")