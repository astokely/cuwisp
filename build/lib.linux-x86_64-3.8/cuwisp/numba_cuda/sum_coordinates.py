from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

from numba import cuda, float64, int32
import numpy as np

def sumCoords(
		coords: np.ndarray, 
		num_traj_frames: int, 
		num_atoms: int, 
		numblocks: int, 
		threadsperblock: int,
) -> np.ndarray:


	@cuda.jit('void(float64[:,:], float64[:,:], int32, int32)')
	def cuSumCoords(v, c, nf, na):
		b = cuda.blockIdx.x
		atomIndex = b
		while atomIndex < na:
			bid = cuda.blockIdx.x
			sm = cuda.shared.array((threadsperblock, 3), float64)
			tid = cuda.threadIdx.x
			bdim = cuda.blockDim.x
			lid = tid
			sm[lid, 0] = 0.0
			sm[lid, 1] = 0.0
			sm[lid, 2] = 0.0
			while lid < nf:
				sm[tid, 0] += v[atomIndex, lid]
				sm[tid, 1] += v[atomIndex, nf + lid]
				sm[tid, 2] += v[atomIndex, 2*nf + lid]
				lid += bdim
			cuda.syncthreads()
			sweep = bdim//2
			while sweep > 0:
				if tid < sweep:
					sm[tid, 0] += sm[tid + sweep, 0]
					sm[tid, 1] += sm[tid + sweep, 1]
					sm[tid, 2] += sm[tid + sweep, 2]
				sweep = sweep//2
				cuda.syncthreads()
			if tid == 0:
				c[atomIndex, 0] = sm[0, 0] / nf
				c[atomIndex, 1] = sm[0, 1] / nf
				c[atomIndex, 2] = sm[0, 2] / nf
			atomIndex += numblocks 
		return
	coords_sum = np.zeros((num_atoms, 3), dtype=np.float64)
	d_coords = cuda.to_device(coords)
	d_coords_sum = cuda.to_device(coords_sum)
	blockDim = threadsperblock
	gridDim = numblocks
	cuSumCoords[gridDim, blockDim](
		d_coords, 
		d_coords_sum, 
		num_traj_frames, 
		num_atoms
	)
	d_coords_sum.copy_to_host(coords_sum)
	return coords_sum


