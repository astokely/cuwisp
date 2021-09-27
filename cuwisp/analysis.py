from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

from .paths import SuboptimalPaths
import numpy as np
from typing import Optional, Dict, List, Tuple
from scipy import interpolate
from scipy.spatial import distance
from math import floor
from multiprocessing import Pool
import numba

def get_frame_index_dict(
		suboptimal_paths: SuboptimalPaths,
) -> Dict:
	frame_index_dict = {
		suboptimal_paths[0][0][0].coordinate_frames[
			frame_index
		] : frame_index for frame_index in
		range(len(
			suboptimal_paths[0][0][0].coordinate_frames
		))	
	}
	return frame_index_dict

def sort_distances(
		distance_vector: np.ndarray,
) -> Dict[int, np.float64]:
	distance_dict = {
		path_index : distance_vector[path_index] 
		for path_index in range(len(distance_vector))
	}
	return {
		path_index : distance 
		for path_index, distance
		in sorted(
			distance_dict.items(), 
			key=lambda distance: distance[1]
		)				
	}

def get_partitions(
		spline: np.ndarray,
		num_partitions: int
) -> Tuple[int]:
	partitions = []
	i = 1
	partition_size = floor(
		len(spline) / num_partitions
	)
	while i <= num_partitions:
		if i == num_partitions:
			partitions.append((
				(i-1) * partition_size,
				len(spline),
			))
		else:
			partitions.append((
				(i-1) * partition_size,
				i * partition_size,
			))
		i += 1
	return partitions

def generate_spline(
		nodes: np.ndarray,
		spline_input_points_incr: Optional[float] = 0.001,
		smoothing_factor: Optional[float] = 0.0,
) -> np.ndarray:
	num_edges = (
		max(nodes.shape) - 1
	)
	degree = num_edges - 1
	if num_edges > 3:
		degree = 3
	x, y, z = nodes
	tck, u = interpolate.splprep(
		[x, y, z],
		s=smoothing_factor,
		k=degree
	)
	u_new = np.arange(
		0,
		1.0 + spline_input_points_incr,
		spline_input_points_incr
	)
	return interpolate.splev(u_new, tck)

def sp_splines(
		suboptimal_paths: SuboptimalPaths,
		frame: int,
		spline_input_points_incr: Optional[float] = 0.01,
		smoothing_factor: Optional[float] = 0.0,
) -> np.ndarray:
	frame = get_frame_index_dict(
		suboptimal_paths
	)[frame]
	splines = []	
	for path in suboptimal_paths.paths:
		spline = np.zeros(
			(path.num_nodes, 3), 
			dtype=np.float64
		)
		spline[0][0] = (
			path.edges[0].node1.coordinates[frame][0]
		)
		spline[0][1] = (
			path.edges[0].node1.coordinates[frame][1]
		)
		spline[0][2] = (
			path.edges[0].node1.coordinates[frame][2]
		)
		for i in range(path.num_edges):
			spline[i+1][0] = (
				path.edges[i].node2.coordinates[frame][0]
			)
			spline[i+1][1] = (
				path.edges[i].node2.coordinates[frame][1]
			)
			spline[i+1][2] = (
				path.edges[i].node2.coordinates[frame][2]
			)
		splines.append(
			np.array(generate_spline(
				spline.T,
				spline_input_points_incr=spline_input_points_incr,
				smoothing_factor=smoothing_factor,
			)).T
		)
	return splines

@numba.jit(nopython=True)
def _frechet_distance(
		a: np.ndarray, 
		i: int, 
		j: int, 
		c1: np.ndarray,
		c2: np.ndarray
) -> np.float64:

	if a[i, j] > -1:
		return a[i, j]
	elif i == 0 and j == 0:
		a[i, j] = (
			np.linalg.norm(c1[i] - c2[j])
		)
	elif i > 0 and j == 0:
		a[i, j] = max(
			_frechet_distance(a, i - 1, 0, c1, c2), 
			np.linalg.norm(c1[i] - c2[j])
		)
	elif i == 0 and j > 0:
		a[i, j] = max(
			_frechet_distance(a, 0, j - 1, c1, c2), 
			np.linalg.norm(c1[i] - c2[j])
		)
	elif i > 0 and j > 0:
		a[i, j] = max(
			min(
				_frechet_distance(a, i - 1, j, c1, c2),
				_frechet_distance(a, i-1, j-1, c1, c2),
				_frechet_distance(a, i, j-1, c1, c2)
			),
			np.linalg.norm(c1[i] - c2[j])
		)
	else:
		a[i, j] = np.inf 
	return a[i, j]


@numba.jit(nopython=True)
def frechet_distance(
		c1: np.ndarray,
		c2: np.ndarray,
) -> np.float64:
	c1_size = c1.shape[0]
	c2_size = c2.shape[0]
	a = (
		np.ones(
			(c1_size, c2_size), 
			dtype=np.float64
		) * -1
	)
	return (
		_frechet_distance(
			a, 
			c1_size - 1, 
			c2_size - 1, 
			c1, 
			c2
		)
	)

def _sp_frechet_distance_vector(
	args: Tuple
) -> np.ndarray:
	(
		reference_path_index,
		splines,
		partition
	) = args
	
	reference_path_spline = splines[
		reference_path_index
	]
	frechet_distance_vector = np.zeros(
		len(splines),
		dtype=np.float64,
	)
	for i in range(len(splines)):
		frechet_distance_vector[i] = (
			frechet_distance(
				reference_path_spline[
					partition[0]:partition[1]
				],
				splines[i][
					partition[0]:partition[1],
				]
			)
		)
	return frechet_distance_vector

def sp_frechet_distance_matrix(
		reference_path_index: int,
		splines: np.ndarray,
		num_partitions: Optional[int] = 1,
		num_multiprocessing_processes: Optional[int] = False,
) -> np.ndarray:
	if not num_multiprocessing_processes:
		num_multiprocessing_processes = num_partitions
	partitions = get_partitions(
		splines[0],
		num_partitions
	)
	reference_path_spline = splines[
		reference_path_index
	]
	frechet_distance_matrix = np.zeros(
		(len(partitions), len(splines)),
		dtype=np.float64,
	)
	parameters = [
		(
			reference_path_index,
			splines,
			partitions[i]
		)
		for i in range(num_partitions)
	]
	with Pool(num_partitions) as pool:
		frechet_distance_vectors = list(pool.map(
			_sp_frechet_distance_vector,
			parameters
		))
	m, n = frechet_distance_matrix.shape
	for i in range(m):
		for j in range(n):
			frechet_distance_matrix[i][j] = (
				frechet_distance_vectors[i][j]
			)
	return frechet_distance_matrix

def _sp_manhattan_distance_vector(
	args: Tuple
) -> np.ndarray:
	(
		reference_path_index,
		splines,
		partition
	) = args
	reference_path_spline = splines[
		reference_path_index
	]
	manhattan_distance_vector = np.zeros(
		len(splines), 
		dtype=np.float64
	) 
	for	index, spline in enumerate(splines):
		manhattan_distances = np.zeros(
			len(reference_path_spline), 
			dtype=np.float64
		) 
		point_index = 0
		for i in range(*partition):
			manhattan_distances[point_index] = (
				distance.cityblock(
					reference_path_spline[i],
					spline[i]	
				)
			)
			point_index += 1
		manhattan_distance_vector[index] = ( 
			np.mean(manhattan_distances)
		)
			
	return manhattan_distance_vector	

def sp_manhattan_distance_matrix(
		reference_path_index: int,
		splines: np.ndarray,
		num_partitions: Optional[int] = 1,
		num_multiprocessing_processes: Optional[int] = False,
) -> np.ndarray:
	if not num_multiprocessing_processes:
		num_multiprocessing_processes = num_partitions
	partitions = get_partitions(
		splines[0],
		num_partitions
	)
	manhattan_distance_matrix = np.zeros(
		(len(partitions), len(splines)),
		dtype=np.float64,
	)
	parameters = [
		(
			reference_path_index,
			splines,
			partitions[i]
		)
		for i in range(num_partitions)
	]
	with Pool(num_partitions) as pool:
		manhattan_distance_vectors = list(pool.map(
			_sp_manhattan_distance_vector,
			parameters
		))
	m, n = manhattan_distance_matrix.shape
	for i in range(m):
		for j in range(n):
			manhattan_distance_matrix[i][j] = (
				manhattan_distance_vectors[i][j]
			)
	return manhattan_distance_matrix

