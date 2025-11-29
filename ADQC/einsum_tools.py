import mlx.core as mx
import numpy as np
import math
from typing import List, Tuple, Optional, Union

def convert_nums_to_abc(nums:List[int], n0:int=0):
    s = ''
    n0 = n0 + 97
    for m in nums:
        s += chr(m + n0)
    return s


def indexes_eq2einsum_eq(indexes:List[List[int]]):
    eq = convert_nums_to_abc(indexes[0])
    for n in range(1, len(indexes)-1):
        eq += (',' + convert_nums_to_abc(indexes[n]))
    eq += ('->' + convert_nums_to_abc(indexes[-1]))
    return eq