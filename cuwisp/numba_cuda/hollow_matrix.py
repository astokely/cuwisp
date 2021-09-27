from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

from abserdes import Serializer as serializer
from numba import cuda, float64, int32, float32, int64
import math
import numpy as np
from time import time


def hollowMatrix(m, h):
	threads_per_block = 256 
	num_blocks = 100
	@cuda.jit('void(float64[:,:], float64[:,:], int32, int32, float64)')
	def cuHollowMatrix(
		m,
		h,
		num_rows,
		num_cols,
		inf,
	):
		bid = cuda.blockIdx.x	
		while bid < num_rows:
			row_min = inf
			for i in range(num_cols):
				row_min = min(row_min, m[bid][i])
			for lid in range(num_cols):
				h[bid][lid] = m[bid][lid] - row_min  
			bid += num_blocks

	num_rows, num_cols = m.shape
	dm = cuda.to_device(m)
	dh = cuda.to_device(h)
	cuHollowMatrix[100, 256](dm, dh, num_rows, num_cols, np.inf)
	dh.copy_to_host(h)
	return h















