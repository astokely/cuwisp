import shutil
import mdtraj as md
import os
import sys
import numpy as np
from scipy.spatial.distance import cdist
from .calccom import calc_com as calc_com
from .cparse import parsePdb as parse
from typing import Any, Optional
from .sum_coordinates import sumCoords
from abserdes import Serializer as serializer
from multiprocessing import Pool
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

	def load_pdb_from_list(self, traj, num_frames=False, frame=False):
		self.trajectory = traj
		top = self.trajectory.topology
		atoms = [atom for atom in top.atoms]
		coords = [10*xyz for xyz in self.trajectory[0].xyz][0]
		if num_frames:
			for i in range(len(coords)):
				scmat[i][frame] = coords[i][0]
				scmat[i][num_frames + frame] = coords[i][1]
				scmat[i][2*num_frames + frame] = coords[i][2]
		
		self.atomnames = np.array([atom.name for atom in atoms]) 
		self.chains = np.array([atom.residue.chain.index for atom in atoms]) 
		self.masses = np.array([atom.element.mass for atom in atoms]) 
		self.resids = np.array([atom.residue.index+1 for atom in atoms])
		self.resnames = np.array([atom.residue.name for atom in atoms])
		self.elements = np.array([atom.element.symbol for atom in atoms])
		self.coordinates = np.array(coords, np.float64)

	def map_atoms_to_residues(self):

		d = {}
		for i in range(len(self.coordinates)):
			d[i] = str(self.chains[i]) + "_" + self.resnames[i] + "_" + str(self.resids[i])
		ident = {} 
		for v in d.values():
			ident[v] = 'a'
		ri = list(ident.keys())
		self.residue_identifiers_in_order = np.array(ri)
		self.residue_identifier_to_atom_indices = {}
		for k, v in d.items():
			if v not in self.residue_identifier_to_atom_indices:
				self.residue_identifier_to_atom_indices[v] = []
			self.residue_identifier_to_atom_indices[v].append(k)
		for k in self.residue_identifier_to_atom_indices.keys():
			self.residue_identifier_to_atom_indices[k] = np.array(self.residue_identifier_to_atom_indices[k])


	def map_nodes_to_residues(self, coms):
		self.nodes_array = np.empty((len(self.residue_identifiers_in_order), 3))
		for index in range(len(coms)):
			self.nodes_array[index][0] = coms[index][0]
			self.nodes_array[index][1] = coms[index][1]
			self.nodes_array[index][2] = coms[index][2]

def parse_pdb(args):
	path, pdb_file, num_frames = args
	frame = int(pdb_file[:-4])
	traj = md.load(path+pdb_file)
	pdb = Molecule()
	pdb.load_pdb_from_list(traj, num_frames, frame)
	return pdb

class GetCorrelationMatrix:

	def __init__(
			self,
			output_directory: str,
			contact_map_distance_limit: float,
			pdb_trajectory_filename: str,
			tmp_path: str,	
			correlation_matrix_filename: str,
			correlation_matrix_after_contact_map_filename: str,
			nodes_xml_filename: str,
	):
		current_frame = 0
		if os.path.exists(tmp_path):
			shutil.rmtree(tmp_path)
		os.makedirs(tmp_path)
		parse(pdb_trajectory_filename, tmp_path + "/")
		num_traj_frames = len(os.listdir(tmp_path + "/")) 
		self.average_pdb = Molecule() 
		self.average_pdb.load_pdb_from_list(md.load(tmp_path + "/0.pdb"))
		pdb_single_frame_files = [f for f in os.listdir(tmp_path + "/")]
		pdbs = []


		num_atoms = len(self.average_pdb.coordinates)
		paths = [tmp_path + "/" for path in range(num_traj_frames)]
		num_frames = [num_traj_frames for path in range(num_traj_frames)]
		global scmat
		cmat = ctypes_matrix(3*num_traj_frames, num_atoms)
		scmat = shared_ctypes_multiprocessing_array(cmat)
		with Pool(50) as pool:
			pdbs = list(pool.map(parse_pdb, zip(paths, pdb_single_frame_files, num_frames)))
		mat = np.ctypeslib.as_array(scmat)
		del scmat
		self.average_pdb.coordinates = sumCoords(mat, num_traj_frames, num_atoms, 1000, 1024)
		self.atom_indices_in_same_node = {}
		top = self.average_pdb.trajectory.topology
		atom_indices_in_nodes = []
		for residue in top.residues:
			atom_indices_in_nodes.append(tuple([atom.index for atom in residue._atoms]))
		for atom_indices in atom_indices_in_nodes:
			for atom_index in atom_indices:
				self.atom_indices_in_same_node[atom_index] = atom_indices
		nodes_dict = {}

		for pdb in pdbs:
			pdb.map_atoms_to_residues()
		self.average_pdb.map_atoms_to_residues()
		nodes = Nodes()
		node_index = 0
		for node_identifier, atom_indices in \
		self.average_pdb.residue_identifier_to_atom_indices.items():
			node = Node()
			node.index = node_index
			node.atom_indices = atom_indices
			node.identifier = node_identifier
			nodes[node_index] = node
			node_index += 1
		nodes.num_nodes = node_index + 1
		if nodes_xml_filename == '':
			nodes_xml_filename = output_directory + "/nodes.xml"
		nodes.serialize(nodes_xml_filename)
		pdbs.append(self.average_pdb)
		all_indices = []
		for pdb in pdbs:
			all_indices.append([
				pdb.residue_identifier_to_atom_indices[residue_iden] 
				for residue_iden in pdb.residue_identifiers_in_order
			])
		
		all_coords = [pdb.coordinates for pdb in pdbs]
		all_masses = [pdb.masses for pdb in pdbs]
		all_coms = calc_com(all_indices, all_coords, all_masses)
		
		for i, pdb in enumerate(pdbs): 
			pdb.map_nodes_to_residues(all_coms[i])
		for pdb in pdbs:
			for index, residue_iden in enumerate(
				pdb.residue_identifiers_in_order
			):
				try:
					nodes_dict[residue_iden].append(pdb.nodes_array[index])
				except:
					nodes_dict[residue_iden] = [pdb.nodes_array[index]]
		dictionary_of_node_lists = nodes_dict
		for res_iden in dictionary_of_node_lists:
			dictionary_of_node_lists[res_iden] = np.array(
				dictionary_of_node_lists[res_iden], np.float64
			)
		set_of_deltas = {}
		res_atoms = {}
		for i in range(len(self.average_pdb.coordinates)):
			if self.average_pdb.resids[i] not in res_atoms:
				res_atoms[self.average_pdb.resids[i]] = []
			res_atoms[self.average_pdb.resids[i]].append(i)
		for index, residue_iden in enumerate(
			self.average_pdb.residue_identifiers_in_order
		):
			set_of_deltas[residue_iden] = (
				dictionary_of_node_lists[residue_iden] 
				- self.average_pdb.nodes_array[index]
			)

		ensmeble_average_deltas_self_dotproducted = {}
		
		for residue_iden in self.average_pdb.residue_identifiers_in_order:
			dot_products = (
				set_of_deltas[residue_iden] * set_of_deltas[residue_iden]
			).sum(axis=1)
			ensmeble_average_deltas_self_dotproducted[residue_iden] = np.average(
				dot_products
			)

		self.correlations = np.empty(
			(
				len(self.average_pdb.residue_identifiers_in_order),
				len(self.average_pdb.residue_identifiers_in_order),
			)
		)

		for x in range(len(self.average_pdb.residue_identifiers_in_order)):
			residue1_key = self.average_pdb.residue_identifiers_in_order[x]
			for y in range(len(self.average_pdb.residue_identifiers_in_order)):
				residue2_key = self.average_pdb.residue_identifiers_in_order[y]

				residue1_deltas = set_of_deltas[residue1_key]
				residue2_deltas = set_of_deltas[residue2_key]

				if len(residue1_deltas) != len(residue2_deltas):
					sys.exit(0)

				dot_products = (residue1_deltas * residue2_deltas).sum(axis=1)

				ensemble_average_dot_products = np.average(dot_products)

				C = ensemble_average_dot_products / np.power(
					ensmeble_average_deltas_self_dotproducted[residue1_key]
					* ensmeble_average_deltas_self_dotproducted[residue2_key],
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
				len(self.average_pdb.residue_identifiers_in_order) - 1
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
					index1 + 1, len(self.average_pdb.residue_identifiers_in_order)
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
