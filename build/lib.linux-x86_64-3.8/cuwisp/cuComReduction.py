'''
By Andy Stokely
'''

from numba import cuda, float64, int32, float32, int64
import math
import numpy as np
import mdtraj as md
import time


def centerOfMassReduction(
		coordinates: np.ndarray,
		masses: np.ndarray, 
		total_masses: np.ndarray,
		threads_per_block: int,
		num_blocks: int,
		node_size: int,
		num_nodes: int,
		num_frames: int, 
) -> np.ndarray:
	@cuda.jit('void(float64[:,:,:], float64[:,:], float64[:,:,:], float64[:], int32, int32)')
	def cudaCenterOfMassReduction(
			coordinates,
			masses, 
			center_of_masses, 
			total_masses,
			num_frames,
			num_iters,
	):
		for node_index in range(num_nodes):
			i = 0
			while i < num_iters:
				sm = cuda.shared.array(
					(threads_per_block, 3), 
					float64
				)
				bid = cuda.blockIdx.x
				tid = cuda.threadIdx.x
				bdim = cuda.blockDim.x
				lid = tid 
				sm[lid][0] = 0.0
				sm[lid][1] = 0.0
				sm[lid][2] = 0.0
				cuda.syncthreads()
				while lid < node_size:
					index = i*num_blocks + bid
					sm[tid, 0] += coordinates[
						node_index, 
						index, 
						lid
					]*masses[node_index, lid]

					sm[tid, 1] += coordinates[
						node_index, 
						index, 
						node_size+lid
					]*masses[node_index,lid]

					sm[tid, 2] += coordinates[
						node_index, 
						index, 
						2*node_size+lid
					]*masses[node_index, lid]
					lid += bdim
				cuda.syncthreads()
				sweep = threads_per_block//2 
				while sweep > 0:
					if tid < sweep:
						sm[tid, 0] += sm[tid + sweep, 0]
						sm[tid, 1] += sm[tid + sweep, 1]
						sm[tid, 2] += sm[tid + sweep, 2]
					cuda.syncthreads()
					sweep = sweep//2
				if tid == 0:
					center_of_masses[
						node_index, 
						bid + i*num_blocks, 
						0
					] = sm[0, 0] / total_masses[node_index]
					center_of_masses[
						node_index, 
						bid + i*num_blocks, 
						1
					] = sm[0, 1] / total_masses[node_index]
					center_of_masses[
						node_index, 
						bid + i*num_blocks, 
						2
					] = sm[0, 2] / total_masses[node_index] 
				cuda.syncthreads()
				i += 1

	padding = math.ceil(num_frames / num_blocks)*num_blocks - num_frames
	num_padded_frames = num_frames + padding
	num_iters = int(num_padded_frames / num_blocks)
	h_center_of_masses = np.array([
		[np.zeros(3, dtype=np.float64) for frame 
		in range(num_padded_frames)] 
			for node in range(num_nodes)
		]
	)
		
	if padding != 0:
		coordinates = np.array([
			np.append(i, [
				np.zeros(3*node_size, dtype=np.float64) for i 
				in range(padding)
			], axis=0) 
			for i in coordinates
		], dtype=np.float64)
	d_coordinates = cuda.to_device(coordinates)
	d_masses = cuda.to_device(masses)
	d_total_masses = cuda.to_device(total_masses)
	d_center_of_masses = cuda.device_array_like(h_center_of_masses)
	cudaCenterOfMassReduction[num_blocks, threads_per_block](
		d_coordinates, 
		d_masses, 
		d_center_of_masses, 
		d_total_masses,
		num_padded_frames,
		num_iters
	)
	d_center_of_masses.copy_to_host(
		h_center_of_masses
	)
	if padding != 0:
		return h_center_of_masses[:,:-padding]
	return h_center_of_masses
