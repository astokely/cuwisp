from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import numpy as np
from numba import cuda

def hollowMatrix(
        m,
        h
        ):
    num_blocks = 100

    @cuda.jit('void(float64[:,:], float64[:,:], int32, int32, float64)')
    def cuHollowMatrix(
            cu_m,
            cu_h,
            cu_num_rows,
            cu_num_cols,
            inf,
    ):
        bid = cuda.blockIdx.x
        while bid < cu_num_rows:
            row_min = inf
            for i in range(cu_num_cols):
                row_min = min(row_min, cu_m[bid][i])
            for lid in range(cu_num_cols):
                cu_h[bid][lid] = cu_m[bid][lid] - row_min
            bid += num_blocks

    num_rows, num_cols = m.shape
    dm = cuda.to_device(m)
    dh = cuda.to_device(h)
    cuHollowMatrix[100, 256](dm, dh, num_rows, num_cols, np.inf)
    dh.copy_to_host(h)
    return h
