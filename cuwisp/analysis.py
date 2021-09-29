from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import os
from .paths import SuboptimalPaths
import numpy as np
from typing import Optional, Dict, List, \
	Tuple, Union
from scipy import interpolate
from scipy.spatial import distance
from math import floor
from multiprocessing import Pool
import numba
import gc
import shutil
from .cfrechet import cFrechet
from abserdes import Serializer as serializer

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

class FrechetDistanceMatrix(serializer):

	def __init__(
			self,
			filename: str,
			reference_path_index: int,
			num_partitions: int,
	) -> None:
		self.filename = filename
		self.reference_path_index = reference_path_index
		self.num_partitions = num_partitions

	@property
	def	matrix(self):
		return np.load(
			f'{self.filename}.npy'
		)

	def ordered(
			self,
			partition_index: int,
	) -> Dict[int, np.float64]:
		return sort_distances(
			self.matrix[partition_index]
		)

class DistanceMatrix(serializer):

	def __init__(
			self,
			filename: str,
			reference_path_index: int,
			num_partitions: int,
	) -> None:
		self.filename = filename
		self.reference_path_index = reference_path_index
		self.num_partitions = num_partitions

	@property
	def	matrix(self):
		return np.load(
			f'{self.filename}.npy'
		)

	def ordered(
			self,
			partition_index: int,
	) -> Dict[int, np.float64]:
		return sort_distances(
			self.matrix[partition_index]
		)
			
class Analysis(serializer):
	
	def __init__(
			self,
			analysis_directory: str,
			frame: int,
			frechet_distance_matrices: \
				Optional[Dict[int, Dict]] = {}, 
			distance_matrices: \
				Optional[Dict[int, Dict]] = {}, 
			splines_directory: Optional[str] = '',
	) -> None:
		self.analysis_directory = (
			analysis_directory
		)
		self.frechet_distance_matrices = (
			frechet_distance_matrices
		)
		self.distance_matrices = (
			distance_matrices
		)
		self.splines_directory = (
			splines_directory
		)
		if analysis_directory:
			if not os.path.exists(analysis_directory):
				os.makedirs(analysis_directory)

	def prepare(
			self,
			suboptimal_paths: SuboptimalPaths,
			spline_input_points_incr: Optional[float] = 0.001,
			smoothing_factor: Optional[float] = 0.0,
	) -> None:
		sp_splines(
			suboptimal_paths,
			self.frame,
			output_directory = (
				f'{self.analysis_directory}/splines'
			),
			spline_input_points_incr = spline_input_points_incr,
			smoothing_factor = smoothing_factor,
		)

	def	distance_matrix(
			self,
			path_index: Optional[int] = 0,
			num_partitions: Optional[int] = 1,
	) -> np.ndarray:
		if not os.path.exists(
			f'{self.analysis_directory}'
			+ f'/distance_matrices'
		):
			os.makedirs(
				f'{self.analysis_directory}'
				+ f'/distance_matrices'
			)
		matrix_filename = (
			 f'{self.analysis_directory}/'
			+ f'distance_matrices/'
			+ f'distance_matrix_{path_index}'
		)
		distance_matrix = sp_distance_matrix(
			path_index,
			f'{self.analysis_directory}/splines',
			num_partitions=num_partitions,
			output_filename=matrix_filename,
		)
		distance_matrix_obj = DistanceMatrix(
 			matrix_filename,
            path_index,
            num_partitions
		)
		self.distance_matrices[path_index] = (
			distance_matrix_obj
		)
		return distance_matrix

	def	frechet_distance_matrix(
			self,
			path_index: Optional[int] = 0,
			num_partitions: Optional[int] = 1,
	) -> np.ndarray:
		if not os.path.exists(
			f'{self.analysis_directory}'
			+ f'/frechet_distance_matrices'
		):
			os.makedirs(
				f'{self.analysis_directory}'
				+ f'/frechet_distance_matrices'
			)
		matrix_filename = (
			 f'{self.analysis_directory}/'
			+ f'frechet_distance_matrices/'
			+ f'frechet_distance_matrix_{path_index}'
		)
		frechet_distance_matrix = (
			sp_frechet_distance_matrix(
				path_index,
				f'{self.analysis_directory}/splines',
				num_partitions=num_partitions,
				output_filename=matrix_filename,
			)
		)
		frechet_distance_matrix_obj = FrechetDistanceMatrix(
 			matrix_filename,
            path_index,
            num_partitions
		)
		self.frechet_distance_matrices[path_index] = (
			frechet_distance_matrix_obj
		)
		return frechet_distance_matrix

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

def generate_spline(
		nodes: np.ndarray,
		spline_input_points_incr,
		smoothing_factor,
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
		output_directory: Optional[str] = False,
		spline_input_points_incr: Optional[float] = 0.001,
		smoothing_factor: Optional[float] = 0.0,
) -> np.ndarray:
	frame = get_frame_index_dict(
		suboptimal_paths
	)[frame]
	splines = []	
	if output_directory:
		if os.path.exists(output_directory):
			shutil.rmtree(output_directory)
		os.makedirs(output_directory)
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
		if output_directory:
			np.save(
				f'{output_directory}/{path.index}.npy', 
				splines[-1]
			)
	return splines

def frechet_distance(
		p: np.ndarray,
		q: np.ndarray,
) -> np.float64:
	p_size = p.shape[0]
	q_size = q.shape[0]
	P = p.T.reshape(3*p_size)
	Q = q.T.reshape(3*q_size)
	ca = np.ones(
		p_size*q_size, 
		dtype=np.float64
	) * -1
	return (
		cFrechet(
			P,
			Q,
			ca,
			p_size-1,
			q_size-1,
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

def load_splines(
		directory: str,
) -> List[np.ndarray]:
	numpy_matrix_files = [
		os.path.abspath(os.path.join(directory, f)) 
		for f in os.listdir(directory) if 'npy' in f
	]
	numpy_matrix_files_dict = {
		int(f.split('/').pop()[:-4]) : f for f in 
		numpy_matrix_files
	} 
	numpy_matrix_files_dict = dict(sorted(
		numpy_matrix_files_dict.items()
	))
	return [
		np.load(f) for f 
		in numpy_matrix_files_dict.values() 
	]
	
		

def sp_frechet_distance_matrix(
		reference_path_index: int,
		splines: Union[np.ndarray, str],
		num_partitions: Optional[int] = 1,
		num_multiprocessing_processes: Optional[int] = False,
		output_filename: Optional[str] = False,
) -> np.ndarray:
	if isinstance(splines, str):
		splines = load_splines(splines)
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
	if output_filename: 
		np.save(
			f'{output_filename}.npy',
			frechet_distance_matrix
		)
	return frechet_distance_matrix

def _sp_distance_vector(
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
	distance_vector = np.zeros(
		len(splines), 
		dtype=np.float64
	) 
	for	index, spline in enumerate(splines):
		distances = np.zeros(
			(partition[1]-partition[0]), 
			dtype=np.float64
		) 
		point_index = 0
		for i in range(*partition):
			distances[point_index] = (
				np.linalg.norm((
					reference_path_spline[i]
					- spline[i]	
				))
			)
			point_index += 1
		distance_vector[index] = ( 
			np.mean(distances)
		)
	gc.collect()
	return distance_vector	

def sp_distance_matrix(
		reference_path_index: int,
		splines: Union[np.ndarray, str],
		num_partitions: Optional[int] = 1,
		num_multiprocessing_processes: Optional[int] = False,
		output_filename: Optional[str] = False,
) -> np.ndarray:
	if isinstance(splines, str):
		splines = load_splines(splines)
	if not num_multiprocessing_processes:
		num_multiprocessing_processes = num_partitions
	partitions = get_partitions(
		splines[0],
		num_partitions
	)
	distance_matrix = np.zeros(
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
		distance_vectors = list(pool.map(
			_sp_distance_vector,
			parameters
		))
	m, n = distance_matrix.shape
	for i in range(m):
		for j in range(n):
			distance_matrix[i][j] = (
				distance_vectors[i][j]
			)
	if output_filename: 
		np.save(
			f'{output_filename}.npy', 
			distance_matrix
		)
	gc.collect()
	return distance_matrix
