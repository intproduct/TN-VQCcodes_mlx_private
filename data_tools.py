import mlx.core as mx
import mlx.nn as nn
import numpy as np
import math
from typing import List, Tuple, Optional

def data_trans(data: mx.array, name: Optional[str] = None) -> mx.array:
    if name is None:
        return data
    elif name == 'mod2pi':
        return mx.array((data + math.pi) % (2 * math.pi) - math.pi)
    elif name == 'softplus':
        return mx.log(1 + mx.exp(data))