#这是关于tensor分解的各种方法
#包括hosvd, ttsvd, tucker, rank one，以及纠缠熵计算

import mlx.core as mx
import numpy as np
import math
from typing import List, Optional, Union, Tuple
from .tensor_ops import tucker_product

def svd_mx(tensor: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
    u, s, vh = mx.linalg.svd(tensor, stream = mx.cpu)
    s0 = s.shape[0]
    if u.shape[-1] > s0:
        u = u[:, :s0]
    elif vh.shape[0] > s0:
        vh = vh[:s0, :]
    return u, s, vh

def reduced_mxtrix(tensor: mx.array, bond: int) -> mx.array:
    #计算bond指标的约化密度矩阵
    index = list(tensor.ndim)
    index.pop(bond)
    shape = tensor.shape
    tensor1 = tensor.transpose([bond] + index).reshape(shape[bond],-1)
    return tensor @ tensor1.T.conj()


def hosvd(tensor: mx.array, dc: Optional[int], return_lms: bool
          ) -> Union[Tuple[mx.array, List[mx.array], List[mx.array]],
                     Tuple[mx.array, List[mx.array]]]:
    if dc is not None:
        dc = [dc] * tensor.ndim
    u, lms = [], []
    for n in range(tensor.ndim):
        rho_m = reduced_mxtrix(tensor, n)
        lm_, u_ = mx.linalg.eigh(rho_m)  #特征值和特征矩阵
        if (dc is not None) and (dc[n] < tensor.shape[n]):
            u_ = u[:, -dc[n]:]
        u.append(u_)
        lms.append(lm_)
    core = tucker_product(tensor, u, dim=0, conj=True)
    if return_lms:
        return core, u, lms
    else:
        return core, u
    
def tt_svd(tensor: mx.array, boundary: str) -> List[mx.array]:
    if boundary == 'open':
        dim_list = list(tensor.shape)
        dimL = 1
        length_tensor = tensor.ndim
        list_u = []
        list_s = []
        list_non_zeros =[]
        for n in range(length_tensor-1):
            T = tensor.reshape(dimL*dim_list[n], -1)
            p,q = T.shape
            ra = mx.random.normal([p,q]) * 1e-10
            t = T + ra 
            u, s, v = svd_mx(tensor)
            u = u.reshape(dimL, dim_list[n], -1)
            list_u.append(u)
            list_s.append(s)
            num = len(s)
            num_non_zeros = len(s[s > 1e-10])
            list_non_zeros.append(num)
            list_non_zeros.append(num_non_zeros)
            dimL = list(s.shape)[0]
            tensor = (mx.diag(s) @ v).unsqueeze(-1)
            #print(f"{n+1}次分解完成")
        list_u.append(tensor)
    elif boundary == 'periodic':
        dim_list = list(tensor.shape)
        dimL = tensor.shape[0]
        length_tensor = tensor.ndim - 2
        list_u = []
        list_s = []
        list_non_zeros =[]
        for n in range(length_tensor-1):
            T = tensor.reshape(dimL*dim_list[n+1], -1)
            p,q = T.shape
            ra = mx.random.normal([p,q])
            t = T + ra
            u, s, v = svd_mx(tensor)
            list_u.append(u)
            list_s.append(s)
            num = len(s)
            num_non_zeros = len(s[s > 1e-10])
            list_non_zeros.append(num)
            list_non_zeros.append(num_non_zeros)
            dimL = list(s.shape)[0]
            tensor = mx.diag(s) @ v
            #print(f"{n+1}次分解完成")
        list_u.append(tensor.reshape(-1, dim_list[-2], dim_list[-1]))

    return list_u

def rank_one_product(vectors: List[mx.array], c: int) -> mx.array:
    x = vectors[0]
    dims = [vectors[0].size]
    for v in vectors[1:]:
        dims.append(v.size)
        x = mx.outer(x, v)
        x = x.reshape(-1)
    return x.reshape(dims) * c

def rank_one(tensor: mx.array, v: Optional[List] = None, it_time: int = 1000, 
             tol: float= 1e-4) -> Tuple[List[mx.array], float]:
    
    ndims = tensor.ndim

    #初始化
    if v is None:
        v= []
        for n in range(tensor.ndim):
            v.append(mx.random.normal(tensor.shape[n]))
    #归一化
    for n in range(tensor.ndim):
        v[n] /= mx.linalg.norm(v[n])

    tensor_indices = "".join(map(chr, range(ord('a'), ord('a') + ndims)))

    norm_1 = 1.0
    err = mx.ones(tensor.ndim)
    err_norm = mx.ones(tensor.ndim)
    for t in range(it_time):
        for n in range(tensor.ndim):
            vec_subs = ",".join(tensor_indices[i] for i in range(ndims) if i != n)
            einsum_lists = f"{tensor_indices},{vec_subs}->{tensor_indices[n]}"
            v_args = [v[i].conj() for i in range(ndims) if i != n]
            x = mx.einsum(einsum_lists, tensor, *v_args)
            norm = mx.linalg.norm(x)
            v1 = x / norm
            err[n] = mx.linalg.norm(v[n]-v1)
            err_norm[n] = mx.abs(norm - norm_1)
            v[n] = v1
            norm_1 = norm
        if mx.max(err) < tol and mx.max(err_norm) < tol:
            break
    
    return v, norm_1

def tucker_rank(tensor: mx.array, eps: float=1e-14) -> List:
    lms = hosvd(tensor, return_lms=True)[2]
    r = []
    for lm in lms:
        r_ = (lm > eps).sum().item()
        r.append(r_)
    return r

def entanglement_entropy(s: mx.array, eps: float=1e-15) -> mx.array:
    prob = s ** 2 + eps
    return mx.inner(-1 * prob, mx.log(prob))