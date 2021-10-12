from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import os
import shutil
import sys
from typing import (
	Optional,
	Tuple,
	List,
	Dict,
)

from .correlation_matrix import get_correlation_matrix
from .paths import (
	get_suboptimal_paths,
	SuboptimalPaths,
	Path,
	Edge,
	Nodes,
	built_in_rules,
	Rule,
)

def calculate_correlation_matrix(
		calculation_name: str,
		output_directory: str,
		contact_map_distance_limit: float,
		trajectory_fname: str,
		cuda_parameters: Optional[Tuple] = (256, 10, 256, 100),
		num_multiprocessing_processes: Optional[int] = 10,
		temp_file_directory: Optional[str] = '',
		topology_fname: Optional[str] = '',
		node_coordinate_frames: Optional[List[int]] = False,
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
		calculation_name,
		output_directory,
		contact_map_distance_limit,
		trajectory_fname,
		topology_fname,
		temp_file_directory,
		threads_per_block_com_calc,
		num_blocks_com_calc,
		threads_per_block_sum_coordinates_calc,
		num_blocks_sum_coordinates_calc,
		num_multiprocessing_processes,
		node_coordinate_frames,
	)

def launch_get_suboptimal_paths(
		parameters: Dict,
) -> None:
	get_suboptimal_paths(**parameters)

def calculate_suboptimal_paths(
		calculation_name: str,
		input_directory: str,
		src: int,
		sink: int,
		cutoff: Optional[float] = False,
		threads_per_block: Optional[int] = 256,
		use_contact_map_correlation_matrix: Optional[bool] = True,
		serialization_frequency: Optional[float] = False,
		simulation_rounds: Optional[List[int]] = False,
		gpu: Optional[int] = 0,
		max_num_paths: Optional[int] = 25,
		path_finding_rules: Optional[Dict] = False,
) -> None:
	if not simulation_rounds:
		simulation_rounds = [0, 1, 2, 3, 4]
	if not path_finding_rules:
		path_finding_rules = {}
	rules = built_in_rules()
	if path_finding_rules:
		for index, rule in path_finding_rules.items():
			rules[index] = Rule(*rule)
	if use_contact_map_correlation_matrix:
		correlation_matrix_fname = (
			f'{input_directory}/'
			f'{calculation_name}_'
			+ f'correlation_matrix_after_contact_map.npy'
		)
	else:
		correlation_matrix_fname = (
			f'{input_directory}/'
			f'{calculation_name}_'
			+ f'correlation_matrix.npy'
		)
	suffix = ''
	for simulation_round in simulation_rounds:
		suffix += f'{simulation_round}_'
	suffix = suffix[:-1]
	nodes_fname = (
		f'{input_directory}/'
		f'{calculation_name}_'
		+ f'nodes.xml'
	)
	apsp_matrix_fname = (
		f'{input_directory}/'
		f'{calculation_name}_'
		+ f'apsp_matrix.npy'
	)
	suboptimal_paths_fname = (
		f'{input_directory}/'
		f'{calculation_name}_'
		+ f'suboptimal_paths.xml'
	)
	suboptimal_paths_fname = (
		f'{suboptimal_paths_fname[:-4]}'
		f'_{suffix}{suboptimal_paths_fname[-4:]}'
	)
	correlation_matrix_serialization_directory = ''
	suboptimal_paths_serialization_directory = ''
	if serialization_frequency:
		correlation_matrix_serialization_directory = (
			f'{input_directory}/'
			+ f'path_finding_serialization/'
			+ f'{calculation_name}_'
			+ f'serialized_correlation_matrices'
		)
		correlation_matrix_serialization_directory = (
			f'{correlation_matrix_serialization_directory}'
			f'_{suffix}'
		)
		if not os.path.exists(
			correlation_matrix_serialization_directory
		):
			os.makedirs(
				correlation_matrix_serialization_directory
			)
		suboptimal_paths_serialization_directory = (
			f'{input_directory}/'
			f'path_finding_serialization/'
			f'{calculation_name}_'
			+ f'serialized_suboptimal_paths'
		)
		suboptimal_paths_serialization_directory = (
			f'{suboptimal_paths_serialization_directory}'
			f'_{suffix}'
		)
		if not os.path.exists(
			suboptimal_paths_serialization_directory
		):
			os.makedirs(
				suboptimal_paths_serialization_directory
			)
	parameters = {
		'correlation_matrix_fname' : (
			correlation_matrix_fname
		),
		'nodes_fname' : nodes_fname,
		'src' : src,
		'sink' : sink,
		'suboptimal_paths_fname' : (
			suboptimal_paths_fname
		),
		'cutoff' : cutoff,
		'threads_per_block' : threads_per_block,
		'serialization_fname' : (
			calculation_name
		),
		'serialization_frequency' : (
			serialization_frequency
		),
		'correlation_matrix_serialization_directory' : (
			correlation_matrix_serialization_directory
		),
		'suboptimal_paths_serialization_directory' : (
			suboptimal_paths_serialization_directory
		),
		'simulation_rounds' : simulation_rounds,
		'gpu_index' : gpu,
		'max_num_paths' : max_num_paths,
		'rules' : rules,
		'apsp_matrix_fname' : apsp_matrix_fname,
	}
	launch_get_suboptimal_paths(parameters)

def get_serialized_suboptimal_paths(
		directory: str,
		rounds: List[int],
) -> List[str]:
	xmls = [
		os.path.abspath(os.path.join(directory, f)) for f
		in os.listdir(directory) if 'xml' in f
	]
	rounds = [
		str(path_finding_round)
		for path_finding_round in rounds

	]
	return list(set([
		xml for xml in xmls
		for path_finding_round in rounds
		if (
				path_finding_round in xml and
				'nodes' not in xml
			)
	]))

def deserialize_suboptimal_paths(
		suboptimal_path_xmls: List[str],
) -> List[SuboptimalPaths]:
	suboptimal_paths = []
	suboptimal_path_xmls = set(suboptimal_path_xmls)
	for xml in suboptimal_path_xmls:
		suboptimal_path = SuboptimalPaths()
		suboptimal_path.deserialize(xml)
		suboptimal_paths.append(suboptimal_path)
	return suboptimal_paths

def get_sorted_suboptimal_paths_dict(
		suboptimal_paths: SuboptimalPaths,
) -> Dict[float, List[Edge]]:
	suboptimal_paths_dict = {
		path.length : path.edges
		for suboptimal_paths_obj
		in suboptimal_paths
		for path in suboptimal_paths_obj.paths
	}
	return dict(sorted(
		suboptimal_paths_dict.items()
	))

def get_src_sink(
	suboptimal_paths: SuboptimalPaths,
) -> Tuple[int]:
	return (
		suboptimal_paths.src,
		suboptimal_paths.sink,
	)

def merge_suboptimal_paths(
		directory: str,
		rounds: List[int],
		nodes_fname: str,
		suboptimal_paths_fname: str,
) -> None:
	suboptimal_paths = (
		get_serialized_suboptimal_paths(
			directory,
			rounds,
		)
	)
	suboptimal_paths_objs = (
		deserialize_suboptimal_paths(
			suboptimal_paths
		)
	)
	src, sink = get_src_sink(
		suboptimal_paths_objs[0]
	)
	suboptimal_paths_dict = (
		get_sorted_suboptimal_paths_dict(
			suboptimal_paths_objs
		)
	)
	suboptimal_paths = SuboptimalPaths()
	nodes_obj = Nodes()
	nodes_obj.deserialize(nodes_fname)
	path_index = 0
	for path_length in suboptimal_paths_dict:
		path = Path()
		path.length = path_length
		path.edges = []
		path_nodes = set([])
		for path_edge in suboptimal_paths_dict[path_length]:
			node1_index = path_edge.node1.index
			node2_index = path_edge.node2.index
			path_nodes.add(node1_index)
			path_nodes.add(node2_index)
			edge = Edge()
			edge.node1 = nodes_obj[node1_index]
			edge.node2 = nodes_obj[node2_index]
			path.edges.append(edge)
		path.src = src
		path.sink = sink
		path.index = path_index
		path.num_nodes = len(path_nodes)
		path.num_edges = len(path.edges)
		suboptimal_paths.paths.append(path)
		path_index += 1
	suboptimal_paths.src = src
	suboptimal_paths.sink = sink
	suboptimal_paths.num_paths = len(suboptimal_paths.paths)
	suboptimal_paths.serialize(suboptimal_paths_fname)
