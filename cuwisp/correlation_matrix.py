from collections import namedtuple
import shutil
import mdtraj as md
import os
import sys
import numpy as np
from scipy.spatial.distance import cdist
from .calccom import calc_com as calc_com
from .cparse import parsePdb as parse
from typing import Any, Optional, Tuple, \
	Union, List, Dict, Mapping
from .sum_coordinates import sumCoords
from abserdes import Serializer as serializer
from multiprocessing import Pool
from collections import defaultdict
from multiprocessing import sharedctypes
from .nodes import Nodes, Node
from .hollow_matrix import hollowMatrix
from .cuda_correlation_matrix import cuda_correlation_matrix
from .cuda_contact_map import cuda_contact_map 
def ctypes_matrix(
		n: int,
		m: int,
) -> Any:
	return np.ctypeslib.as_ctypes(np.array([
		np.ctypeslib.as_ctypes(np.zeros(n, dtype=np.float64)) 
		for i in range(m)
	]))

def shared_ctypes_multiprocessing_array(
		ctypes_array: Any,
) -> Any:
	return sharedctypes.RawArray(
		ctypes_array._type_, 
		ctypes_array
	)

class Molecule:

	def load_pdb_from_list(
				self, 
				traj: md.core.trajectory, 
				num_frames: Optional[bool] = False, 
				frame: Optional[bool] = False,
	) -> None:
		self.trajectory = traj
		top = self.trajectory.topology
		atoms = [atom for atom in top.atoms]
		coords = [10*xyz for xyz in self.trajectory[0].xyz][0]
		if num_frames:
			for i in range(len(coords)):
				scmat[i][frame] = coords[i][0]
				scmat[i][num_frames + frame] = coords[i][1]
				scmat[i][2*num_frames + frame] = coords[i][2]
		
		self.atomnames = np.array([
			atom.name for atom in atoms
		]) 
		self.chains = np.array([
			atom.residue.chain.index for atom in atoms
		]) 
		self.masses = np.array([
			atom.element.mass for atom in atoms
		]) 
		self.resids = np.array([
			atom.residue.index for atom in atoms
		])
		self.resnames = np.array([
			atom.residue.name for atom in atoms
		])
		self.segment_ids = np.array([
			atom.residue.segment_id for atom in atoms
		])
		self.elements = np.array([
			atom.element.symbol for atom in atoms
		])
		self.coordinates = np.array(coords, np.float64)

	def map_atoms_to_node_tags(self) -> None:
		indices_tags_dict = {
			index : (
				f'{self.chains[index]}_'
				+ f'{self.resnames[index]}_'
				+ f'{self.resids[index]}_'
				+ f'{self.segment_ids[index]}'
			) 
			if self.segment_ids[index] != '' else (
				f'{self.chains[index]}_'
				+ f'{self.resnames[index]}_'
				+ f'{self.resids[index]}'
			) 
			for index in range(len(self.coordinates))
		}
		self.node_tags_in_order = np.array(list({ 
			tag : None 
			for tag in 
			indices_tags_dict.values()
		}))
		self.node_tag_to_atom_indices = defaultdict(list)
		for atom_index, tag in indices_tags_dict.items():
			self.node_tag_to_atom_indices[tag].append(
				atom_index
			)		 

	def map_nodes_to_node_tags(
			self, 
			coms: np.ndarray,
	) -> None:
		self.nodes_array = np.empty((
			len(self.node_tags_in_order), 
			3
		))
		for index in range(len(coms)):
			self.nodes_array[index][0] = coms[index][0]
			self.nodes_array[index][1] = coms[index][1]
			self.nodes_array[index][2] = coms[index][2]

def parse_pdb(
		args: Tuple[Union[int, int, str]],
) -> Molecule:
	path, pdb_file, num_frames = args
	frame = int(pdb_file[:-4])
	traj = md.load(path+pdb_file)
	pdb = Molecule()
	pdb.load_pdb_from_list(traj, num_frames, frame)
	return pdb

def prepare_trajectory_for_analysis(
		temp_file_directory: str,
		pdb_trajectory_filename: Optional[str] = '',
		dcd_trajectory_filename: Optional[str] = '',
		topology_filename: Optional[str] = '',
) -> List[str]:
	if os.path.exists(temp_file_directory):
		shutil.rmtree(temp_file_directory)
	os.makedirs(temp_file_directory)
	if pdb_trajectory_filename != '':
		parse(
			pdb_trajectory_filename, 
			temp_file_directory + "/"
		)
	else:
		index = 0
		for frame in md.iterload(
			dcd_trajectory_filename,
			top=topology_filename,
			chunk = 1,
		):
			frame.save_pdb(
				f'{temp_file_directory}/{index}.pdb'
			)
			index += 1
	pdb_single_frame_files = [
		pdb_file for pdb_file 
		in os.listdir(temp_file_directory + "/")
	]
	print(pdb_single_frame_files)
	return pdb_single_frame_files

def get_parameters_for_multiprocessing_pdb_parser(
		temp_file_directory: str,
		pdb_from_trajectory: Molecule,
) -> Tuple[Union[int, int, List[str], int]]:
	num_traj_frames = len(
		os.listdir(temp_file_directory + "/")
	) 
	num_atoms = len(
		pdb_from_trajectory.coordinates
	)
	paths = [
		temp_file_directory + "/" 
		for path in range(num_traj_frames)
	]
	num_frames = [
		num_traj_frames for path	
		in range(num_traj_frames)
	]
	return (
		num_traj_frames,
		num_atoms,
		paths,
		num_frames,
	)

def multiprocessing_pdb_parser(
		num_traj_frames: int,
		num_atoms: int,
		num_multiprocessing_processes: int,
		paths: List[str],
		pdb_single_frame_files: List[str],
		num_frames: int,
		average_pdb: Molecule,
		num_blocks_sum_coordinates_calc: int, 
		threads_per_block_sum_coordinates_calc: int,
) -> Molecule:
	global scmat
	cmat = ctypes_matrix(
		3*num_traj_frames, 
		num_atoms
	)
	scmat = shared_ctypes_multiprocessing_array(cmat)
	with Pool(num_multiprocessing_processes) as pool:
		pdbs = list(pool.map(
			parse_pdb, 
			zip(
				paths, 
				pdb_single_frame_files, 
				num_frames
			)
		))
	coordinates = np.ctypeslib.as_array(scmat)
	del scmat
	average_pdb.coordinates = sumCoords(
		coordinates,
		num_traj_frames, 
		num_atoms, 
		num_blocks_sum_coordinates_calc, 
		threads_per_block_sum_coordinates_calc
	)
	for pdb in pdbs:
		pdb.map_atoms_to_node_tags()
	average_pdb.map_atoms_to_node_tags()
	pdbs.append(average_pdb)
	return pdbs

def serialize_nodes(
		average_pdb: Molecule,
		coms: np.ndarray,
		output_directory: str,
		nodes_xml_filename: str,
) -> None:
	nodes = Nodes()
	node_index = 0
	atom_indices_list = []
	for node_tag, atom_indices in \
	average_pdb.node_tag_to_atom_indices.items():
		node = Node()
		node.index = node_index
		node.atom_indices = atom_indices
		node.tag = node_tag
		node.segment_id = (
			f'{average_pdb.segment_ids[atom_indices[0]]}' 
		)
		node.resname = (
			f'{average_pdb.resnames[atom_indices[0]]}' 
		)
		node.chain_index = (
			average_pdb.chains[atom_indices[0]] 
		)
		node.resid = (
			average_pdb.resids[atom_indices[0]]  
		)
		nodes[node_index] = node
		node_index += 1
		nodes[node.index].coordinates = (
			coms[-1][node.index]
		)
		atom_indices_list.append(atom_indices)
	nodes.num_nodes = node_index + 1
	if nodes_xml_filename == '':
		nodes_xml_filename = (
			f'{output_directory}/nodes.xml'
		)
	nodes.serialize(nodes_xml_filename)
	return atom_indices_list 
	
def calculate_center_of_masses(	
		pdbs: List[Molecule],
		threads_per_block_com_calc: int, 
		num_blocks_com_calc: int,
) -> np.ndarray:
	all_indices = []
	for pdb in pdbs:
		all_indices.append([
			pdb.node_tag_to_atom_indices[node_tag] 
			for node_tag 
			in pdb.node_tags_in_order
		])
	
	all_coords = [
		pdb.coordinates for pdb in pdbs
	]
	all_masses = [
		pdb.masses for pdb in pdbs
	]
	all_coms = calc_com(
		all_indices, 
		all_coords, 
		all_masses, 
		threads_per_block_com_calc, 
		num_blocks_com_calc
	)
	return all_coms

def save_matrix(
		output_directory: str,	
		matrix_filename: str,
		default_matrix_filename: str,
		numpy_array: np.ndarray,
) -> None:
	if matrix_filename == '':
		matrix_filename = (
			f'{output_directory}/{default_matrix_filename}'
		)
	else:
		matrix_filename = (
			f'{output_directory}/{matrix_filename}'
		)
	np.savetxt(
		matrix_filename,
		numpy_array,
	)

def save_average_pdb(
		output_directory: str,
		average_pdb_filename: str,
		default_average_pdb_filename: str,
		average_pdb: Molecule,
) -> None:
	if average_pdb_filename == '':
		average_pdb_filename = (
			f'{output_directory}/{default_average_pdb_filename}'
		)
	else:
		average_pdb_filename = (
			f'{output_directory}/{average_pdb_filename}'
		)
	average_pdb.trajectory.save_pdb(
		average_pdb_filename
	)	
		
def get_node_com_coordinates_array(
			num_nodes: int,
			num_pdbs: int,
			pdbs: List[Molecule],
) -> np.ndarray:
	nodes = np.array(
		[
			np.zeros((num_pdbs, 3), dtype=np.float64)
			for node in range(num_nodes) 
		]
		,dtype=object
	) 
	for pdb_index in range(num_pdbs):
		for node_index in range(num_nodes):
			nodes[node_index][pdb_index][0] = (
				pdbs[pdb_index].nodes_array[node_index][0]
			)
			nodes[node_index][pdb_index][1] = (
				pdbs[pdb_index].nodes_array[node_index][1]
			)
			nodes[node_index][pdb_index][2] = (
				pdbs[pdb_index].nodes_array[node_index][2]
			)
	return nodes

def get_contact_map(
		correlation_matrix: np.ndarray,
		average_pdb: Molecule,
		contact_map_distance_limit: float,
		node_atom_indices: List,
		num_nodes: int,
) -> np.ndarray:
	contact_map = np.ones(correlation_matrix.shape)
	avg_coords = average_pdb.coordinates
	correlation_matrix_after_contact_map = np.zeros(
		correlation_matrix.shape,
		dtype=np.float64
	)
	if contact_map_distance_limit != np.inf:
		cuda_contact_map(
			correlation_matrix, 
			correlation_matrix_after_contact_map, 
			contact_map, 
			contact_map_distance_limit, 
			avg_coords, 
			node_atom_indices
		)
		return contact_map, correlation_matrix_after_contact_map
	return contact_map, correlation_matrix

def _get_correlation_matrix(
		num_nodes: int,
		nodes,
		average_pdb
) -> np.ndarray:
	correlation_matrix = np.zeros((
		num_nodes,
		num_nodes
	), dtype=np.float64)
	h_correlation_matrix = np.zeros((
		num_nodes,
		num_nodes
	), dtype=np.float64)
	cuda_correlation_matrix(
		nodes,
		average_pdb.nodes_array,
		correlation_matrix
	)
	hollowMatrix(
		correlation_matrix,
		h_correlation_matrix
	)
	return h_correlation_matrix

def get_correlation_matrix(
		output_directory: str,
		contact_map_distance_limit: float,
		trajectory_filename: str,
		topology_filename: str,
		temp_file_directory: str,	
		correlation_matrix_filename: str,
		correlation_matrix_after_contact_map_filename: str,
		nodes_xml_filename: str,
		threads_per_block_com_calc: int,
		num_blocks_com_calc: int,
		threads_per_block_sum_coordinates_calc: int,
		num_blocks_sum_coordinates_calc: int,
		num_multiprocessing_processes: int,
) -> None:
	current_frame = 0
	if trajectory_filename[-3:] == 'pdb':
		pdb_single_frame_files = prepare_trajectory_for_analysis(
			temp_file_directory,
			pdb_trajectory_filename = trajectory_filename
		)
	elif trajectory_filename[-3:] == 'dcd':
		pdb_single_frame_files = prepare_trajectory_for_analysis(
			temp_file_directory,
			dcd_trajectory_filename = trajectory_filename,
			topology_filename = topology_filename
		)
	else:
		raise Exception
	average_pdb = Molecule() 
	average_pdb.load_pdb_from_list(
		md.load(f'{temp_file_directory}/0.pdb')
	)
	num_traj_frames, num_atoms, paths, num_frames = (
		get_parameters_for_multiprocessing_pdb_parser(
			temp_file_directory,
			average_pdb
		)
	) 
	pdbs = multiprocessing_pdb_parser(
		num_traj_frames,
		num_atoms,
		num_multiprocessing_processes,
		paths,
		pdb_single_frame_files,
		num_frames,
		average_pdb,
		num_blocks_sum_coordinates_calc, 
		threads_per_block_sum_coordinates_calc,
	)
	shutil.rmtree(temp_file_directory)
	average_pdb = pdbs[-1]
	coms = calculate_center_of_masses(
		pdbs,
		threads_per_block_com_calc, 
		num_blocks_com_calc,
	)
	for i, pdb in enumerate(pdbs): 
		pdb.map_nodes_to_node_tags(coms[i])
	node_atom_indices = (
		serialize_nodes(
			average_pdb,
			coms,
			output_directory,
			nodes_xml_filename,
		)
	)

	num_nodes = len(node_atom_indices)
	num_pdbs = len(pdbs)
	nodes = get_node_com_coordinates_array(
		num_nodes, 
		num_pdbs,
		pdbs,
	)
	correlation_matrix = (
		_get_correlation_matrix(
			num_nodes,
			nodes.astype(np.float64),
			average_pdb,
		)
	)
	save_matrix(
		output_directory,
		correlation_matrix_filename,
		'correlation_matrix.txt',
		correlation_matrix,
	)

	contact_map, correlation_matrix = get_contact_map( 
		correlation_matrix,
		average_pdb,
		contact_map_distance_limit,
		node_atom_indices,
		num_nodes,
	)

	save_matrix(
		output_directory,
		correlation_matrix_after_contact_map_filename,
		'correlation_matrix_after_contact_map.txt',
		correlation_matrix,
	)

	save_matrix(
		output_directory,
		'',
		'contact_map.txt',
		contact_map,
	)
		
	save_average_pdb(
		output_directory,
		'',
		'average.pdb',
		average_pdb,
	)
