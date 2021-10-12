from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import numpy as np
from numba import cuda

def hollowMatrix(
        m: np.ndarray,
        h: np.ndarray,
) -> np.ndarray:
    """
    @param m:
    @type m: numpy.ndarray

    @param h:
    @type h: numpy.ndarray

    @return:
    @rtype: numpy.ndarray

    """
    num_blocks = 100

    @cuda.jit
    def cuHollowMatrix(
            cu_m,
            cu_h,
            cu_num_rows,
            cu_num_cols,
            inf,
    ):
        """
        @param cu_m:
        @type cu_m: numpy.ndarray

        @param cu_h:
        @type cu_h: numpy.ndarray

        @param cu_num_rows:
        @type cu_num_rows: int

        @param cu_num_cols:
        @type cu_num_cols: int

        @param inf:
        @type inf: float

        @return:
        @rtype: None

        """
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
