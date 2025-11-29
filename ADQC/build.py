import mlx.core as mx
from .gates import *
from .ADGate import *

@ADgate.register_gate('hadamard', 'h')
def _(self: ADgate):
    return hadamard()

@ADgate.register_gate('cnot')
def _(self: ADgate):
    return cnot()

@ADgate.register_gate('cz')
def _(self: ADgate):
    return cz()

@ADgate.register_gate('s', 's_gate')
def _(self: ADgate):
    return s_gate()

@ADgate.register_gate('t', 't_gate')
def _(self: ADgate):
    return t_gate()

@ADgate.register_gate('swap')
def _(self: ADgate):
    return swap()

@ADgate.register_gate('toffoli')
def _(self: ADgate):
    return toffoli()

@ADgate.register_gate('pauli_ops')
def _(self: ADgate):
    return pauli_ops(self.spec.other_params['name'])

@ADgate.register_gate('delta')
def _(self: ADgate):
    dim = self.get("dim")
    order = self.get("order")
    dtype = self.spec.dtype
    return super_diag_tensor(dim,order, dtype)

@ADgate.register_gate('spin_ops')
def _(self: ADgate):
    return spin_ops(self.spec.other_params['j'])

#以下都是参数门
@ADgate.register_gate('crz')
def _(self: ADgate):
    return crz(self.params['theta'])

@ADgate.register_gate('rotate_ops')
def _(self: ADgate):
    return rotate_ops(self.params)

@ADgate.register_gate('phase_shift')
def _(self: ADgate):
    return phase_shift(self.params['theta'])

@ADgate.register_gate('spin_rotate')
def _(self: ADgate):
    return spin_rotate(self.spec.other_params['j'], self.spec.other_params['theta'], self.params['n'])

@ADgate.register_gate('latent_gate', 'latent')
def _(self: ADgate):
    return latent_gate(self.spec.position, self.spec.qubit_dims, self.spec.other_params['init_way'], self.spec.dtype)

#一个说不上来是参数门还是不是参数门的家伙/////////这个写法现在存疑！！！！！！
@ADgate.register_gate('arbitrary', 'arbitrary_gate')
def _(self: ADgate):
    return arbitrary_gate(self.spec.qubit_dims, self.spec.other_params['matrix'])