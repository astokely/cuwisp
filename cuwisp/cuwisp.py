from multiprocessing import Pool
import gc
import shutil
import mdtraj as md
import os
import sys
import numpy as np
from typing import Optional, Tuple, Union, List, Dict
from .correlation_matrix import get_correlation_matrix 
from .paths import get_suboptimal_paths

def calculate_correlation_matrix(
		output_directory: str,
		contact_map_distance_limit: float,
		trajectory_filename: str,
		correlation_matrix_filename: Optional[str] = '',
		correlation_matrix_after_contact_map_filename: Optional[str] = '',
		nodes_xml_filename: Optional[str] = '',
		cuda_parameters: Optional[Tuple] = (256, 10, 256, 100),
		num_multiprocessing_processes: Optional[int] = 10,
		temp_file_directory: Optional[str] = '',
		topology_filename: Optional[str] = ''
) -> None:
	if temp_file_directory == '':
		temp_file_directory = (
			os.path.dirname(os.path.abspath(
				sys.modules[get_correlation_matrix.__module__].__file__
			)) + "/tmp"
		)
	if os.path.exists(output_directory):
		shutil.rmtree(output_directory)
	os.makedirs(output_directory)
	(
		threads_per_block_com_calc,
		num_blocks_com_calc,
		threads_per_block_sum_coordinates_calc,
		num_blocks_sum_coordinates_calc,
	) = cuda_parameters

	get_correlation_matrix(
		output_directory,
		contact_map_distance_limit,
		trajectory_filename,
		topology_filename,
		temp_file_directory,
		correlation_matrix_filename,
		correlation_matrix_after_contact_map_filename,
		nodes_xml_filename,
		threads_per_block_com_calc,
		num_blocks_com_calc,
		threads_per_block_sum_coordinates_calc,
		num_blocks_sum_coordinates_calc,
		num_multiprocessing_processes,
	)

def launch_get_suboptimal_paths(
		parameters: Dict,
) -> None:
	get_suboptimal_paths(**parameters)

def calculate_suboptimal_paths(
		input_files_path: str,
		src: int,
		sink: int,
		cutoff: Optional[float] = None,
		threads_per_block: Optional[int] = 256,
		use_contact_map_correlation_matrix: Optional[bool] = True,
		correlation_matrix_filename: Optional[str] = '',
		nodes_xml_filename: Optional[str] = '',
		suboptimal_paths_xml_filename: Optional[str] = '',
		serialization_filename: Optional[str] = '',
		serialization_frequency: Optional[float] = 1,
		correlation_matrix_serialization_directory: Optional[str] = '',
		suboptimal_paths_serialization_directory: Optional[str] = '', 
		simulation_rounds: Optional[List[int]] = [0, 1, 2, 3, 4],
		serialization_index: Optional[int] = 0,
		max_edges: Optional[int] = 0,
		gpu: Optional[Union[int, List[int]]] = 0,
) -> None:
		if isinstance(gpu, int):
			gpu = [gpu] 
		if not isinstance(simulation_rounds[0], list):
			simulation_rounds = [simulation_rounds] 
		args = []
		for i in range(len(gpu)):
			if correlation_matrix_filename == '':
				if use_contact_map_correlation_matrix: 
					correlation_matrix_file = (
						"correlation_matrix_after_contact_map.txt" 
					)
				else:
					correlation_matrix_file = "correlation_matrix.txt" 
			if nodes_xml_filename == '':
				nodes_xml_file = "nodes.xml"
			if suboptimal_paths_xml_filename == '':
				suboptimal_paths_xml_filename = "suboptimal_paths.xml" 
			if serialization_filename:
				serialization_suffix = f''
				for simulation_round in simulation_rounds[i]:
					serialization_suffix += f'{simulation_round}_'
				serialization_suffix = serialization_suffix[:-1]
				if correlation_matrix_serialization_directory == '':
					correlation_matrix_serialization_directory = (
						f'serialized_correlation_matrices'
					)
				correlation_matrix_serialization_directory = (
					f'{correlation_matrix_serialization_directory}'
					f'_{serialization_suffix}'
				)
				if not os.path.exists(
					correlation_matrix_serialization_directory
				):
					os.makedirs(
						correlation_matrix_serialization_directory
					)
				if suboptimal_paths_serialization_directory == '':
					suboptimal_paths_serialization_directory = (
						f'serialized_suboptimal_paths'
					)
				suboptimal_paths_serialization_directory = (
					f'{suboptimal_paths_serialization_directory}'
					f'_{serialization_suffix}'
				)
				if not os.path.exists(
					suboptimal_paths_serialization_directory
				):
					os.makedirs(
						suboptimal_paths_serialization_directory
					)
			parameters = {
				'input_files_path' : input_files_path, 
				'correlation_matrix_file' : (
					correlation_matrix_file
				),
				'nodes_xml_file' : nodes_xml_file,
				'src' : src, 
				'sink' : sink,
				'suboptimal_paths_xml' : (
					suboptimal_paths_xml_filename 
				),
				'cutoff' : cutoff,
				'threads_per_block' : threads_per_block,
				'serialization_filename' : (
					serialization_filename
				),
				'serialization_frequency' : (
					serialization_frequency
				),
				'correlation_matrix_serialization_path' : (
					correlation_matrix_serialization_directory
				),
				'suboptimal_paths_serialization_path' : (
					suboptimal_paths_serialization_directory
				),
				'simulation_rounds' : simulation_rounds[i],
				'serialization_index' : serialization_index,
				'max_edges' : max_edges,
				'gpu_index' : gpu[i],
			}
			args.append(parameters)
		with Pool(len(gpu)) as pool:
			pool.map(launch_get_suboptimal_paths, args)



