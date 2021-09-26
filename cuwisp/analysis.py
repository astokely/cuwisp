from .paths import SuboptimalPaths
import numpy as np
from typing import Optional, Dict, List, Tuple
from scipy import interpolate
from scipy.spatial import distance
from math import floor
import numba

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

def sp_average_manhattan_distances(
		splines: np.ndarray,
) -> Dict[int, np.float64]:	
	ssp = splines[0]
	average_manhattan_distances = {}
	for	index, spline in enumerate(splines):
		manhattan_distances = np.zeros(
			len(splines), 
			dtype=np.float64
		) 
		for point_index in range(len(splines)):
			manhattan_distances[point_index] = (
				distance.cityblock(
					ssp[point_index], 
					spline[point_index]
				)
			)
		average_manhattan_distances[index] = ( 
			np.mean(manhattan_distances)
		)
		average_manhattan_distances = {
			path_index : average_manhattan_distance 
			for path_index, average_manhattan_distance 
			in sorted(
				average_manhattan_distances.items(), 
				key=lambda item: item[1]
			)
		}
	return average_manhattan_distances	


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

def sp_frechet_distances(
		splines: np.ndarray,
		path_index: Optional[int] = 0,
		num_partitions: Optional[int] = 1,
) -> Dict[int, np.float64]:
	path = splines[path_index]
	partitions = get_partitions(
		path,
		num_partitions
	)
	frechet_distances = {}
	for index, partition in enumerate(partitions):
		_frechet_distances = {}
		for path_index in range(len(splines)):
			_frechet_distances[path_index] = (
				frechet_distance(
					path[
						partition[0]:partition[1]
					],
					splines[path_index][
						partition[0]:partition[1]
					]
				)
			)	
		_frechet_distances = {
			path_index : frechet_distance_ 
			for path_index, frechet_distance_ 
			in sorted(
				_frechet_distances.items(), 
				key=lambda item: item[1]
			)
		}
		frechet_distances[(partition)] = (
			_frechet_distances
		)
	return frechet_distances


def sp_frechet_distance_matrices(
		splines: np.ndarray,
		num_partitions: Optional[int] = 1,
) -> Dict[Tuple, np.ndarray]:
	partitions = get_partitions(
		splines[0],
		num_partitions
	)
	frechet_distance_matrices = {}
	for partition in partitions:
		frechet_distance_matrix = np.zeros(
			(len(splines), len(splines)),
			dtype=np.float64,
		)
		for i in range(len(splines)):
			path = splines[i]
			for j in range(len(splines)):
				frechet_distance_matrix[i][j] = (
					frechet_distance(
						splines[i][
							partition[0]:partition[1]
						],
						splines[j][
							partition[0]:partition[1],
						]
					)
				)
		frechet_distance_matrices[partition] = (
			frechet_distance_matrix
		)
	return frechet_distance_matrices
				












