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
from .build import *
from .ADGate import *

from typing import List, Tuple, Optional, Union, Dict, Any, Callable


class ADQC(nn.Module):
    def __init__(self, spec):
        super.__init__()
        #self.device
        self.single_gate = True
        self.dtype = mx.complex64
        self.layers = nn.Sequential()

    def act_nth_gate(self, state, n):
        m_p = len(self.layers[n].spec.position)
        m_c = len(self.layers[n].spec.control)
        n_qubits = state.ndim
        shape = state.shape
        perm = list(range(n_qubits))
        for pp in self.layers[n].spec.position:
            perm.remove(pp)
        for pp in self.layers[n].spec.control:
            perm.remove(pp)
        perm = self.layers[n].spec.position + self.layers[n].spec.control
        state1 = state.transpose(perm).reshape(2 ** m_p, 2 ** m_c)
        state1_ = state1[:,:,:-1]
        state2_ = self.layers[n].tensor.reshape(-1, 2 ** m_p) @ state1[:,:,-1]
        state1 = mx.concatenate([state1_, state2_.reshape(state2_.shape + [1,])], axis=-1)
        state = state1.reshape(shape)
        return state1.transpose(inverse_permu(perm))
    
    def act_nth_muti_states_gate(self, states, n):
        m_p = len(self.layers[n].spec.position)
        m_c = len(self.layers[n].spec.control)
        n_qubits = states.ndim - 1
        shape = states.shape
        state1 = states.transpose(list(range(1, n_qubits+1)) + [0])
        perm = list(range(n_qubits))
        for pp in self.layers[n].spec.position:
            perm.remove(pp)
        for pp in self.layers[n].spec.control:
            perm.remove(pp)
        perm = self.layers[n].spec.position + self.layers[n].spec.control
        state1 = state.transpose(perm + [n_qubits]).reshape(2 ** m_p, -1, 2 ** m_c, shape[0])
        state1_ = state1[:,:,:-1,:]
        state2_ = mx.einsum('ab, bcn -> acn', self.layers[n].tensor.shape(-1, 2 ** m_p), state1[:,:,-1,:])
        s_ = state2_.shape
        state2_ = state2_.reshape(s_[0], s_[1], 1, s_[2])
        state1 = mx.concatenate([state1_, state2_], axis=2)
        state = state1.reshape(shape[1:] + (shape[0],))
        perm1 = [m+1 for m in perm] + [0]
        return state1.transpose(inverse_permu(perm))
    
    @staticmethod
    def act_single_gate(state, gate, position, control):
        m_p = len(position)
        m_c = len(control)
        n_qubits = state.ndim
        shape = state.shape
        perm = list(range(n_qubits))
        for pp in position:
            perm.remove(pp)
        for pp in control:
            perm.remove(pp)
        perm = position + perm + control
        state1 = state.transpose(perm).reshape(2 ** m_p, -1, 2 ** m_c)
        state1_ = state1[:,:,:-1]
        state2_ = gate.reshape(-1, 2 ** m_p) @ state1[:,:,-1]
        state1 = mx.concatenate([state1_, state2_.reshape(state2_.shape + [1,])], axis=-1)
        state = state1.reshape(shape)
        return state1.transpose(inverse_permu(perm))
    
    @staticmethod
    def act_single_ADgate(state, gate):
        m_p = len(gate.spec.position)
        m_c = len(gate.spec.control)
        n_qubits = state.shape
        shape = state.shape
        perm = len(range(n_qubits))
        for pp in gate.spec.position:
            perm.remove(pp)
        for pp in gate.spec.control:
            perm.remove(pp)
        perm = gate.spec.position + perm + gate.spec.control
        state1 = state.transpose(perm).reshape(2 ** m_p, -1, 2 ** m_c)
        state1_ = state1[:,:,:-1]
        state2_ = gate.tensor.reshape(-1, 2 ** m_p) @ state1[:,:,-1]
        state1 = mx.concatenate([state1_, state2_.reshape(state2_.shape + [1,])], axis=-1)
        state = state1.reshape(shape)
        return state1.transpose(inverse_permu(perm))
    
    def add_ADgates(self, gates: List[ADgate], name: Optional[str] = None):
        for x in gates:
            if name is None:
                name = str(len(self.layers)) + '_' + x.name
            self.layers.add_module(name, x)

    def __call__(self, state):
        self.renew_gates()
        if self.single_state:
            for n in range(len(self.layers)):
                state = self.act_nth_gate(state, n)
        else:
            for n in range(len(self.layers)):
                state = self.act_nth_muti_states_gate(state, n)
        return state
        
    def renew_gates(self):
        for n in range(len(self.layers)):
            self.layers[n].renew_gate()


class ADQC_LatentGate(ADQC):
    """继承自ADQC，使用隐门（lantent gate）分解避免了幺正约束的复杂性"""
    def __init__(self, pos_one_layer: Optional[List[int]], lattice: Optional[str] = 'brick',
                 num_q: Optional[int] = 10, depth: Optional[int] = 3, 
                 init_way: Optional[str] = 'random',
                 dtype: mx.Dtype = mx.complex64):
        super().__init__(dtype = dtype)
        self.lattice = lattice
        self.depth = depth
        self.init_way = init_way
        if pos_one_layer is None:
            self.position = position_one_layer(self.lattice, num_q)
        else:
            self.position = pos_one_layer
        
        tensor: Optional[mx.array] = None
        for nd in range(len(self.depth)):
            for ng in range(len(self.position)):
                if self.init_way == "identity":
                    tensor = mx.random.normal((4,4)) * 1e-8 + mx.eye(4)
                    tensor = tensor.astype(self.dtype)
                name = self.lattice + '_layer' + str(nd) + '_gate' + str(ng)
                latent_gate = GateSpec(
                    'latent_gate',
                    'parameter',
                    [2,2,2,2],
                    self.position[ng],
                    
                )
                gate = ADgate(latent_gate)
                self.layers.add_module(name, gate)

class VQC(ADQC):

    def __init__(self, pos_one_layer=None, state_type='tensor', lattice='brick',
                 num_q=10, depth=3, ini_way='random',
                 device=None, dtype=mx.complex64):
        super().__init__(state_type=state_type, device=device, dtype=dtype)
        self.lattice = lattice.lower()
        self.depth = depth
        self.ini_way = ini_way
        if pos_one_layer is None:
            self.pos = position_one_layer(self.lattice, num_q) # list of [pos1, pos2]
        else:
            self.pos = pos_one_layer

        R_poses = list(range(num_q))
        X_even = list(range(0, num_q, 2))
        X_odd = list(range(1, num_q-1, 2))
        X_poses = [X_even, X_odd]
        for nd in range(depth):
            for direction in ['z1', 'y', 'z2']:
                self.add_Ri_gate(nd, R_poses, direction, ini_way=self.ini_way)
            for ng in X_poses[nd%2]:
                name = self.lattice + '_layer' + str(nd) + '_CNOT' + str(ng)
                vqc_gate = GateSpec(
                    'x',
                    'parameters',
                    [2,2,2,2],
                    ng+1,
                    ng,
                )
                #gate = ADgate(
                #    'x', pos=ng+1, pos_control=ng,
                #    device=self.device, dtype=self.dtype)
                gate = ADgate(vqc_gate)
                self.layers.add_module(name, gate)

    def add_Ri_gate(self, nd, R_poses, direction, ini_way='identity'):
        '''Add parameterized rotation gate initialized near the identity matrix to the module'''
        for R_pos in R_poses:
            name = 'layer{:d}_R({}){:d}'.format(int(nd), direction, int(R_pos))
            para = mx.random.normal(1)*0.01 if ini_way=='identity' else mx.random.normal(1)
            ri_gate = GateSpec(
                'rotate_'+direction[0],
                'parameters',
                [2,2,2,2],
                R_pos
            )
            #gate = ADGate(
            #                'rotate_'+direction[0], pos=R_pos, paras=para,
            #                device=self.device, dtype=self.dtype
            #            )
            gate = ADgate(ri_gate)
            self.layers.add_module(name, gate)

def act_single_ADgate(state: mx.array, gate: ADgate):
    m_p = len(gate.pos)
    m_c = len(gate.pos_control)
    n_qubit = state.ndimension()
    shape = state.shape
    perm = list(range(n_qubit))
    for pp in gate.pos:
        perm.remove(pp)
    for pp in gate.pos_control:
        perm.remove(pp)
    perm = gate.pos + perm + gate.pos_control
    state1 = state.permute(perm).reshape(2 ** m_p, -1, 2 ** m_c)
    state1_ = state1[:, :, :-1]
    state2_ = gate.tensor.reshape(-1, 2 ** m_p).mm(state1[:, :, -1])
    state1 = mx.concatenate([state1_, state2_.reshape(state2_.shape + (1,))], dim=-1)
    state1 = state1.reshape(shape)
    return state1.permute(inverse_permu(perm))

def get_diff_tensor(g: ADgate, pos_diff):
    gate = GateSpec(
        'unitary',
        'parameters',
        [2,2,2,2],
        g.spec.pos,
        [pos_diff] + g.spec.control,
        mx.complex64,
        g.tensor.shape
    )
    return ADgate(gate)

def position_one_layer(pattern, num_q):
    pos = list()
    if pattern == 'stair':
        for m in range(num_q-1):
            pos.append([m,m+1])
    else:
        m=0
        while m < num_q-1:
            pos.append([m,m+1])
            m+=2
        m=1
        while m < num_q-1:
            pos.append([m,m+1])
            m+=2
    return pos