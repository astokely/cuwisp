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
	Union, List, Dict
from .sum_coordinates import sumCoords
from abserdes import Serializer as serializer
from multiprocessing import Pool
from collections import defaultdict
from multiprocessing import sharedctypes
from .nodes import Nodes, Node

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
			atom.residue.index+1 for atom in atoms
		])
		self.resnames = np.array([
			atom.residue.name for atom in atoms
		])
		self.elements = np.array([
			atom.element.symbol for atom in atoms
		])
		self.coordinates = np.array(coords, np.float64)

	def map_atoms_to_residues(self) -> None:

		d = {}
		atom_indices_to_identifiers_map = {
			index : (
				f'{self.chains[index]}_'
				+ f'{self.resnames[index]}_'
				+ f'{self.resids[index]}'
			) 
			for index in range(len(self.coordinates))
		}
		self.residue_identifiers_in_order = np.array(list({ 
			identifier : None 
			for identifier in 
			atom_indices_to_identifiers_map.values()
		}))
		self.residue_identifier_to_atom_indices = defaultdict(list)
		for k, v in atom_indices_to_identifiers_map.items():
			self.residue_identifier_to_atom_indices[v].append(k)		 


	def map_nodes_to_residues(
			self, 
			coms: np.ndarray,
	) -> None:
		self.nodes_array = np.empty((len(self.residue_identifiers_in_order), 3))
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
		pdb_trajectory_filename: str,
) -> List[str]:
	if os.path.exists(temp_file_directory):
		shutil.rmtree(temp_file_directory)
	os.makedirs(temp_file_directory)
	parse(pdb_trajectory_filename, temp_file_directory + "/")
	pdb_single_frame_files = [
		pdb_file for pdb_file 
		in os.listdir(temp_file_directory + "/")
	]
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
		pdb.map_atoms_to_residues()
	average_pdb.map_atoms_to_residues()
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
	for node_identifier, atom_indices in \
	average_pdb.residue_identifier_to_atom_indices.items():
		node = Node()
		node.index = node_index
		node.atom_indices = atom_indices
		node.identifier = node_identifier
		nodes[node_index] = node
		node_index += 1
		nodes[node.index].coordinates = (
			coms[-1][node.index]
		)
	nodes.num_nodes = node_index + 1
	if nodes_xml_filename == '':
		nodes_xml_filename = (
			f'{output_directory}/nodes.xml'
		)
	nodes.serialize(nodes_xml_filename)
	
def calculate_center_of_masses(	
		pdbs: List[Molecule],
		threads_per_block_com_calc: int, 
		num_blocks_com_calc: int,
) -> np.ndarray:
	all_indices = []
	for pdb in pdbs:
		all_indices.append([
			pdb.residue_identifier_to_atom_indices[residue_iden] 
			for residue_iden 
			in pdb.residue_identifiers_in_order
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

def numpyify_dict(
		d: Dict,
		dtype,
) -> np.ndarray:
	return {
		key : np.array(value, dtype=dtype)
		for key, value in d.items()
	}
	

class GetCorrelationMatrix:

	def __init__(
			self,
			output_directory: str,
			contact_map_distance_limit: float,
			pdb_trajectory_filename: str,
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
		pdb_single_frame_files = prepare_trajectory_for_analysis(
			temp_file_directory,
			pdb_trajectory_filename
		)
		self.average_pdb = Molecule() 
		self.average_pdb.load_pdb_from_list(
			md.load(temp_file_directory + "/0.pdb")
		)
		num_traj_frames, num_atoms, paths, num_frames = (
			get_parameters_for_multiprocessing_pdb_parser(
				temp_file_directory,
				self.average_pdb
			)
		) 
		self.atom_indices_in_same_node = {}
		pdbs = multiprocessing_pdb_parser(
			num_traj_frames,
			num_atoms,
			num_multiprocessing_processes,
			paths,
			pdb_single_frame_files,
			num_frames,
			self.average_pdb,
			num_blocks_sum_coordinates_calc, 
			threads_per_block_sum_coordinates_calc,
		)
		self.average_pdb = pdbs[-1]
		coms = calculate_center_of_masses(
			pdbs,
			threads_per_block_com_calc, 
			num_blocks_com_calc,
		)
		for i, pdb in enumerate(pdbs): 
			pdb.map_nodes_to_residues(coms[i])
		serialize_nodes(
			self.average_pdb,
			coms,
			output_directory,
			nodes_xml_filename,
		)



		nodes = defaultdict(list) 
		for pdb in pdbs:
			for index, residue_iden in enumerate(
				pdb.residue_identifiers_in_order
			):
				nodes[residue_iden].append(
					pdb.nodes_array[index]
				)
		
		num_nodes = len(self.average_pdb.residue_identifiers_in_order)
		numpyify_dict(nodes, np.float64)
		node_indices = [i for i in range(len(nodes))]	

		new_nodes = {}
		new_id_to_atom = {}
		for i in node_indices:
			new_nodes[i] = nodes[list(nodes)[i]]
		for i in node_indices:
			new_id_to_atom[i] = (
				self.average_pdb.residue_identifier_to_atom_indices[
					list(self.average_pdb.residue_identifier_to_atom_indices)[i]
				]
			)
		self.average_pdb.residue_identifier_to_atom_indices = new_id_to_atom
		nodes = new_nodes
		self.average_pdb.residue_identifiers_in_order = list(nodes.keys())

		set_of_deltas = [] 
		for index in range(num_nodes):
			set_of_deltas.append((
				nodes[index] 
				- self.average_pdb.nodes_array[index]
			))
		ensmeble_average_deltas_self_dotproducted = np.zeros(
			len(self.average_pdb.residue_identifiers_in_order),
			dtype=np.float64 
		)
		for index in range(num_nodes):
			dot_products = (
				set_of_deltas[index] * set_of_deltas[index]
			).sum(axis=1)
			ensmeble_average_deltas_self_dotproducted[index] = np.average(
				dot_products
			)

		self.correlations = np.empty(
			(
				num_nodes,
				num_nodes
			)
		)
		for x in range(num_nodes):
			for y in range(num_nodes):

				residue1_deltas = set_of_deltas[x]
				residue2_deltas = set_of_deltas[y]

				if len(residue1_deltas) != len(residue2_deltas):
					sys.exit(0)

				dot_products = (residue1_deltas * residue2_deltas).sum(axis=1)
				ensemble_average_dot_products = np.average(dot_products)

				C = ensemble_average_dot_products / np.power(
					ensmeble_average_deltas_self_dotproducted[x]
					* ensmeble_average_deltas_self_dotproducted[y],
					0.5,
				)

				self.correlations[x][y] = -np.log(
					np.fabs(C)
				)  
		np.savetxt(
			output_directory + "/correlation_matrix.txt",
			self.correlations,
		)

		contact_map = np.ones(self.correlations.shape)
		if contact_map_distance_limit != np.inf:
			for index1 in range(
				num_nodes-1
			):
				residue_iden1 = self.average_pdb.residue_identifiers_in_order[
					index1
				]
				residue1_pts = self.average_pdb.coordinates[
					self.average_pdb.residue_identifier_to_atom_indices[
						residue_iden1
					]
				]
				for index2 in range(
					index1 + 1, num_nodes 
				):
					residue_iden2 = self.average_pdb.residue_identifiers_in_order[
						index2
					]
					residue2_pts = self.average_pdb.coordinates[
						self.average_pdb.residue_identifier_to_atom_indices[
							residue_iden2
						]
					]
					min_dist_between_residue_atoms = np.min(
						cdist(residue1_pts, residue2_pts)
					)
					if (
						min_dist_between_residue_atoms
						> contact_map_distance_limit
					):
						# so they are far apart
						self.correlations[index1][index2] = np.inf 
						self.correlations[index2][index1] = np.inf 
						contact_map[index1][index1] = 0.0
						contact_map[index2][index1] = 0.0

		if correlation_matrix_filename == '':
			correlation_matrix_filename = (
				output_directory
				+ "/correlation_matrix.txt"
			)
		np.savetxt(
			output_directory + "/contact_map_matrix.txt", contact_map
		)

		self.average_pdb.trajectory.save_pdb(
			output_directory + "/average.pdb"
		)

		if correlation_matrix_after_contact_map_filename == '':
			correlation_matrix_after_contact_map_filename = (
				output_directory
				+ "/correlation_matrix_after_contact_map.txt"
			)
		np.savetxt(
			correlation_matrix_after_contact_map_filename,
			self.correlations,
		)


	def __getitem__(self, _):
		return self.correlations
