import shutil
import mdtraj as md
import os
import sys
import numpy as np
from typing import Optional, Tuple
from .correlation_matrix import GetCorrelationMatrix
from .paths import get_suboptimal_paths


def calculate_correlation_matrix(
		output_directory: str,
		contact_map_distance_limit: float,
		pdb_trajectory_filename: str,
		correlation_matrix_filename: Optional[str] = '',
		correlation_matrix_after_contact_map_filename: Optional[str] = '',
		nodes_xml_filename: Optional[str] = '',
		cuda_parameters: Optional[Tuple] = (256, 10, 256, 100),
		num_multiprocessing_processes: Optional[int] = 10,
		temp_file_directory: Optional[str] = '',
) -> None:
	if temp_file_directory == '':
		temp_file_directory = (
			os.path.dirname(os.path.abspath(
				sys.modules[GetCorrelationMatrix.__module__].__file__
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

	correlation_matrix = GetCorrelationMatrix(
		output_directory,
		contact_map_distance_limit,
		pdb_trajectory_filename,
		temp_file_directory,
		correlation_matrix_filename,
		correlation_matrix_after_contact_map_filename,
		nodes_xml_filename,
		threads_per_block_com_calc,
		num_blocks_com_calc,
		threads_per_block_sum_coordinates_calc,
		num_blocks_sum_coordinates_calc,
		num_multiprocessing_processes
	)
	shutil.rmtree(temp_file_directory)

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
		serialization_xml_filename: Optional[
			str
		] = '',
		serialization_frequency: Optional[
			int
		] = 1,
) -> None:
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
		get_suboptimal_paths(
			input_files_path, 
			correlation_matrix_file,
			nodes_xml_file,
			src, 
			sink,
			suboptimal_paths_xml_filename, 
			cutoff,
			threads_per_block,
			serialization_xml_filename,
			serialization_frequency,
		)



