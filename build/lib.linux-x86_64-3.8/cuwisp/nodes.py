from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import re
from abserdes import Serializer as serializer
from collections import namedtuple
from typing import List, Optional
import numpy as np
import os

class Node(serializer):

	def __init__(
			self,
			index: Optional[int] = None,
			atom_indices: Optional[List] = None,
			resname: Optional[str] = None,
			chain_index: Optional[int] = None,
			resid: Optional[int] = None,
			segment_id: Optional[str] = None,
			tag: Optional[str] = None,
			coordinates_directory: Optional[str] = None,
			xml_directory: Optional[str] = False,
			coordinate_frames: Optional[List[int]] = None,	
	):
		self.index = index 
		self.atom_indices = atom_indices 
		self.resname = resname 
		self.resid = resid
		self.chain_index = chain_index 
		self.tag = tag 
		self.coordinates_directory = coordinates_directory 
		self.segment_id = segment_id,
		self.xml_directory = xml_directory
		self.coordinate_frames = coordinate_frames

	def __repr__(self):
		repr_namedtuple = namedtuple(
			f'Node', 
			'index tag resname chain_index resid segment_id'
		)
		return str(repr_namedtuple(
			self.index, 
			self.tag,
			self.resname,
			self.chain_index,
			self.resid,
			self.segment_id,
		))  
	@property
	def coordinates(
			self,
	) -> np.ndarray:
		if self.xml_directory:
			self.coordinates_directory = (
				f'{self.xml_directory}/node_coordinates'
			)
		coordinates_directory = (
			f'{self.coordinates_directory}'
		)
		coordinate_files = [
			os.path.abspath(os.path.join(coordinates_directory, f)) 
			for f in os.listdir(coordinates_directory)
		]
		coordinate_file_frames_dict = {}
		for coordinate_file in coordinate_files: 
			coordinate_file_frame = (
				int(re.findall(
					'\d+', coordinate_file.rsplit('/', 1).pop()
				).pop())
			)
			coordinate_file_frames_dict[coordinate_file_frame] = (
				coordinate_file
			)
		coordinates = []
		for frame in self.coordinate_frames:
			coordinates.append(np.load(
				coordinate_file_frames_dict[frame])[self.index]
			)
		return np.array(coordinates)


class Nodes(serializer):

	def __init__(self):
		self.num_nodes = 0
		self.nodes = []
		return

	def __iter__(self):
		for node in self.nodes:
			yield node

	def __setitem__(
			self, 
			index: int,
			node: Node,
	) -> None:
		if index >= len(self.nodes):
			self.nodes.append(node)
		else:
			self.nodes[index] = node

	def __getitem__(
			self, 
			index: int,
	) -> Node:
		if isinstance(index, slice):
			if index.stop != None:
				if abs(index.stop) < len(self.nodes):
					return self.nodes[index.start:index.stop]
			else:
				if abs(index.start) < len(self.nodes):
					return self.nodes[index.start:index.stop]
		if abs(index) < len(self.nodes):
			return self.nodes[index]
		return

	def __len__(self):
		return len(self.nodes)

	def get_node_from_atom_index(
			self, 
			index: int,
	) -> Node:
		for node in self.nodes:
			if index in node.atom_indices:
				return node

	def find_nodes(
			self, 
			chain_index: Optional[int] = None,
			resid: Optional[int] = None,
			segment_id: Optional[str] = None,
			resname: Optional[str] = None
	) -> List:
		tag = ''
		if chain_index is not None:
			tag += f'{chain_index}_'
		if resid is not None:
			tag += f'{resid}_'
		if segment_id is not None:
			tag += f'{segment_id}_'
		if resname is not None:
			tag += f'{resname}_'
		tag = tag[:-1]
		nodes = []
		for node in self.nodes:
			node_tag = ''
			if chain_index is not None:
				node_tag += f'{node.chain_index}_'
			if resid is not None:
				node_tag += f'{node.resid}_'
			if segment_id is not None:
				node_tag += f'{node.segment_id}_'
			if resname is not None:
				node_tag += f'{node.resname}_'
			node_tag = node_tag[:-1]
			if node_tag == tag:
				nodes.append(node)
		return nodes




