# 对tensor的不同操作
# 包括tucker积，交换逆序，kron积

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from typing import List, Tuple, Optional

def kron(mats: List[mx.array]) -> mx.array:
    assert len(mats) >= 2
    mat = mx.kron(mats[0], mats[1])
    for i in range(2, len(mats)):
        mat = mx.kron(mat, mats[i])
    return mat

def inverse_permu(permu: List[int]) -> List[int]:
    perm = mx.array(permu)
    inv_perm = mx.zeros_like(perm)
    inv_perm[perm] = mx.arange(perm.size(0))
    return inv_perm.tolist()

def tucker_product(tensor: mx.array, ops: List, pos: Optional[List[int]], dim: int, conj: bool) -> mx.array:
    #完成Tucker积操作，Y=G\times U_1\times U_2...\times U_n，不同的U与core tensor G不同指标收缩
    #反过来理解，pos是变得，ops的序列是固定的
    #即第k个ops矩阵作用到第pos[k]指标上
    if pos is None:
        assert len(ops) == tensor.ndim
        ind = list(range(len(ops)))
    legs = list(range(tensor.ndim))
    for k in range(len(pos)):
        pos_now = legs.index(pos[k])
        if conj:
            tensor = mx.tensordot(tensor, ops[k].conj(), [[pos_now], [dim]])
            #注意tensordot计算完成后会把计算的指标放到最后，然后其他指标依次提前
        else:
            tensor = mx.tensordot(tensor, ops[k], [[pos_now], [k]])
    p = ind.pop(pos_now) #先移除位置
    ind += [p] #再把位置添加到最后，保持形状和位置指标一致
    order = inverse_permu(ind)
    return tensor.transpose(order) #mlx的转置函数使用transpose，和numpy一致

def permute_dim(l: List, perm: List[int]) -> List:
    #将一个列表按照perm顺序重排
    return [l[k] for k in perm]