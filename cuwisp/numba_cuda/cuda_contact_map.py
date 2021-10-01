from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import numpy as np
from numba import cuda
from math import sqrt
from math import ceil 
from typing import Tuple


def cuda_contact_map(
	correlation_matrix: np.ndarray, 
	correlation_matrix_after_contact_map: np.ndarray, 
	contact_map: np.ndarray, 
	distance_cutoff: float, 
	average_pdb_coordinates: np.ndarray, 
	node_atom_indices: np.ndarray,
) -> Tuple[np.ndarray]:

	tpb = 32 

	@cuda.jit
	def contact_map_kernel(
			correlation_matrix, 
			contact_map, 
			distance_cutoff, 
			average_pdb_coordinates, 
			node_atom_indices,
			node_atom_indices_offsets,
			inf,
	) -> None:
		i, j = cuda.grid(2)
		if i < (correlation_matrix.shape[0] - 1) and j < correlation_matrix.shape[0]:
			minn = inf
			for x in range(
				node_atom_indices_offsets[i], 
				node_atom_indices_offsets[i+1]
			):
				for y in range(
					node_atom_indices_offsets[j], 
					node_atom_indices_offsets[j+1]
				):
					i1 = node_atom_indices[x]
					i2 = node_atom_indices[y]
					x1 = average_pdb_coordinates[i1][0]
					y1 = average_pdb_coordinates[i1][1]
					z1 = average_pdb_coordinates[i1][2]
					x2 = average_pdb_coordinates[i2][0]
					y2 = average_pdb_coordinates[i2][1]
					z2 = average_pdb_coordinates[i2][2]
					distance = sqrt(
						(x1 - x2)**2
						+ (y1 - y2)**2
						+ (z1 - z2)**2
					)
					minn = min(distance, minn)
			if minn > distance_cutoff:
				contact_map[i][j] = 0.0 
				contact_map[j][i] = 0.0 
				correlation_matrix[i][j] = inf 
				correlation_matrix[j][i] = inf 

	d_correlation_matrix = cuda.to_device(
		correlation_matrix
	)
	d_contact_map = cuda.to_device(contact_map)
	h_node_atom_indices_offsets = np.zeros(
		correlation_matrix.shape[0] +1, 
		dtype=np.int64
	) 
	h_node_atom_indices_offsets[0] = 0
	index= 1
	offset = 0  
	for atom_indices in node_atom_indices:
		offset += len(atom_indices)
		h_node_atom_indices_offsets[index] = offset
		index += 1
	d_node_atom_indices_offsets = cuda.to_device(
		h_node_atom_indices_offsets
	)
	h_node_atom_indices = np.zeros(
		average_pdb_coordinates.shape[0], 
		dtype=np.int64
	) 
	index = 0
	for atom_indices in node_atom_indices:
		for atom_index in atom_indices:
			h_node_atom_indices[index] = atom_index
			index +=  1
	d_node_atom_indices = cuda.to_device(
		h_node_atom_indices
	)
	d_average_pdb_coordinates = cuda.to_device(
		average_pdb_coordinates
	)
	distance_cutoff = np.float64(
		distance_cutoff
	)
	inf = np.float64(np.inf)

	threadsperblock = (tpb, tpb)
	blockspergrid= (
		ceil(correlation_matrix.shape[0] / threadsperblock[0]),
		ceil(correlation_matrix.shape[1] / threadsperblock[1]),
	)
	contact_map_kernel[blockspergrid, threadsperblock](
		d_correlation_matrix, 
		d_contact_map, 
		distance_cutoff, 
		d_average_pdb_coordinates, 
		d_node_atom_indices,
		d_node_atom_indices_offsets,
		inf
	)
	d_contact_map.copy_to_host(contact_map)
	d_correlation_matrix.copy_to_host(
		correlation_matrix_after_contact_map
	)
	return (
		contact_map, 
		correlation_matrix_after_contact_map
	)












