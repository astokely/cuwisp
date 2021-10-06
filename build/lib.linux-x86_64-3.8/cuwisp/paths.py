from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import time
import mdtraj as md
from collections import deque, defaultdict
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from numba import cuda
import numpy as np
import os
import math
import cupy as cp
import inspect
from typing import Any, Tuple, Optional, \
	List, Union, Set, Deque, Callable, \
	Dict 
from math import ceil
from math import floor
from abserdes import Serializer as serializer
from .nodes import Nodes, Node
from .numba_cuda.hedetniemi import hedetniemi_distance

class Rule(object):

	def __init__(
			self, 
			append: Callable,
			pop: Callable, 
	) -> None:
		self._pop = pop
		self._append = append 

	def __repr__(self):
		try:
			return (
				f'{inspect.getsource(self._append)}\n'
				+ f'{inspect.getsource(self._pop)}'
			)
		except:
			return str(type(self)) 

	def pop(
			self,
			dq: Deque,
	) -> Any:
		return self._pop(dq)

	def append(
			self,
			dq: Deque,
			val
	) -> None:
		return self._append(dq, val)

class SuboptimalPaths(serializer):

	def __init__(self) -> None:
		self.paths = []
		self.src = None
		self.sink = None
		self.num_paths = None
		return

	def __iter__(self):
		for path in self.paths:
			yield path

	def __repr__(self):
		suboptimal_paths = ''
		for path in self.paths:
			path = f'{path}\n'
			suboptimal_paths += path
		return suboptimal_paths

	def __getitem__(self, index):
		if index < len(self.paths):
			return self.paths[index]

	def __setitem__(self, index, path):
		if index >= len(self.paths):
			self.paths.append(path)
		else:
			selt.paths[index] = path

	def __len__(self):
		return len(self.paths)

	def reverse(self):
		for path in reversed(self.paths):
			yield path

	def find_paths_with_node(
			self,
			node_index: int
	) -> List:
		path_indices_with_node = []
		for path in self.paths:
			path_node_indices = [
				node_index for node_indices in [
					(
						edge.node1.index,
						edge.node2.index,
					) for edge in path.edges
				]
				for node_index in node_indices
			]
			if node_index in path_node_indices:
				path_indices_with_node.append(
					path
				)
		return path_indices_with_node

	def update(
			self,
			paths: List,
	) -> None:
		self.paths = self.paths + paths
		paths_dict = {
			path.length : path for path in self.paths 
		}
		self.paths = list(dict(sorted(
			paths_dict.items()
		)).values())
		self.num_paths = len(self.paths)

	def factory(
			self,
			path_indices: List,
	) -> Any:
		path_indices = path_indices
		suboptimal_paths = SuboptimalPaths()
		for index in path_indices:
			suboptimal_paths.paths.append(
				self.paths[index]
			)
		suboptimal_paths.src = self.src 
		suboptimal_paths.sink = self.sink 
		suboptimal_paths.num_paths = len(
			suboptimal_paths.paths
		) 
		return suboptimal_paths


class Path(serializer):

	def __init__(self) -> None:
		self.index = None
		self.num_nodes = None
		self.num_edges = None
		self.src = None
		self.sink = None
		self.edges = [] 
		self.length = None

	def __repr__(self):
		edges = self.edges
		length = self.length
		return f'{edges}: {length}' 

	def __iter__(self):
		for edge in self.edges:
			yield edge 

	def __getitem__(self, index):
		if index < len(self.edges):
			return self.edges[index]

	def __setitem__(self, index, edge):
		if index >= len(self.edges):
			self.edges.append(edge)
		else:
			selt.edges[index] = edge

	def __len__(self):
		return len(self.edges)

	def resname_count(
			self,
			reverse: Optional[bool] = True,
	) -> Dict:
		resname_count_dict = defaultdict(int)
		for edge in self.edges:
			resname_count_dict[
				edge.node1.resname
			] += 1
			resname_count_dict[
				edge.node2.resname
			] += 1
		return {
			resname : count for resname, count in sorted(
				resname_count_dict.items(),
				key=lambda items : items[1],
				reverse=reverse
			)
		} 

	@property
	def swap_src_sink(self) -> None:
		rev_edges = []
		src = self.sink
		sink = self.src
		for edge in reversed(self.edges):
			rev_edge = Edge()
			rev_edge.node1 = edge.node2
			rev_edge.node2 = edge.node1
			rev_edges.append(rev_edge)
		self.edges = rev_edges
		self.src = src
		self.sink = sink 
		
 

	def get_edge_index_from_node_index(
			self,
			node_index: int,
			node_edge_pos: int,
	) -> int:
		edge_index = 0
		edge_index_dict = {}
		for edge in self.edges:
			if node_edge_pos == 0:
				edge_index_dict[
					edge.node1.index
				] = edge_index
			else:
				edge_index_dict[
					edge.node2.index
				] = edge_index
			edge_index += 1	
		return edge_index_dict[node_index]

	def get_common_src_sink(
			self,
			path: Any,
	) -> int:
		if self.src == path.src:
			return 0 
		elif self.sink == path.sink:
			return 1 
		return 2

	def factory(
			self,
			path: Any,
			correlation_matrix: np.ndarray,
	) -> Any:
		new_path = Path()
		new_path.src = self.src
		new_path.sink = self.sink
		common_src_sink = self.get_common_src_sink(
			path
		)
		if common_src_sink == 2:
			path.swap_src_sink
			common_src_sink = self.get_common_src_sink(
				path
			)
		if common_src_sink == 0:
			for edge in path.edges:
				new_path.edges.append(edge)
			new_path.edges = (
				new_path.edges
			 	+ self.edges[slice(*(
					self.get_edge_index_from_node_index(
						path.sink,
						common_src_sink,
					),
					self.num_edges
				))]	
			)
		else:
			new_path.edges = self.edges[slice(*(
				0,
				self.get_edge_index_from_node_index(
					path.src,
					common_src_sink,
				) + 1
			))]
			for edge in path.edges:
				new_path.edges.append(edge)
		new_path.num_edges = len(new_path.edges)
		new_path.num_nodes = new_path.num_edges + 1
		new_path.length = np.float64(0.0)
		for edge in new_path.edges:
			new_path.length += correlation_matrix[
				edge.node1.index,	
				edge.node2.index,	
			]
		new_path.index = 0
		return new_path

class Edge(serializer):

	def __init__(self) -> None:
		self.node1 = Node() 
		self.node2 = Node() 
	
	def __repr__(self) -> str:
		return str((
			self.node1.index, 
			self.node2.index
		))

	def __iter__(self):
		nodes = [self.node1, self.node2]
		for node in nodes:
			yield node 

	def __getitem__(self, index):
		if index == 0:
			return self.node1
		elif index == 1:
			return self.node2

	def __setitem__(self, index, node):
		if index == 0:
			self.node1 = node
		elif index == 1:
			self.node2 = node

	def __len__(self):
		return 2 

def ordered_paths(
		paths: List,
		src: int,
) -> List:
	ordered_paths = []
	paths = list(paths)
	pos = None
	while paths:
		for i in reversed(paths):
			if src in i:
				ordered_paths.append(i)
				pos = i[1]
				paths.remove(i)
			elif i[0] == pos:
				ordered_paths.append(i)
				pos = i[1]
				paths.remove(i)
	return ordered_paths

def append_middle(
		dq: deque, 
		val: Tuple,
) -> None:
	middle_index = floor(len(dq)/2)
	dq.insert(middle_index, val)

def pop_middle(
		dq: deque,
) -> Tuple:
	middle_index = floor(len(dq)/2)
	middle_val = dq[middle_index]
	del dq[middle_index]
	return middle_val

def append(
		dq: deque, 
		val: Tuple,
) -> None:
	dq.append(val)
	return

def pop(
		dq: deque,
) -> Tuple:
	return dq.pop()

def append_left(
		dq: deque, 
		val: Tuple,
) -> None:
	dq.appendleft(val)
	return

def pop_left(
		dq: deque,
) -> Tuple:
	return dq.popleft()

def built_in_rules() -> Dict[int, Rule]:
	return {
		0 : Rule(append, pop),
		1 : Rule(append_middle, pop_middle),
		2 : Rule(append, pop_left),
		3 : Rule(append_middle, pop_left),
		4 : Rule(append_middle, pop),
	}

def get_ssp(
		src: int, 
		sink: int, 
		h: np.ndarray,
		a: np.ndarray,
) -> Tuple:
	pos = sink 
	p = h[src][sink] 
	h_row = h[:,src]
	a_col = a[sink,:]
	path = []
	nodes = []
	nodes.append(pos)
	while pos != src:
		h_row_sorted_indices = cp.argsort(
			h_row
		)
		closest = {} 
		found_next_node = False
		for i in h_row_sorted_indices:
			if h_row[i] == np.inf:
				break
			dist = h_row[i] + a_col[i]
			if i != pos:
				if p == dist == np.inf:
					return 
				closest[abs(dist-p)] = i 
			if dist == p and i != pos:
				prev_pos = pos 
				pos = i
				a_col = a[pos,:]
				path.append((pos, prev_pos))
				nodes.append(pos)
				p = h_row[pos] 
				found_next_node = True
				break
		if not found_next_node:
			minn = min(closest.keys())
			prev_pos = pos 
			pos = closest[minn]
			a_col = a[pos,:]
			path.append((pos, prev_pos))
			nodes.append(pos)
			p = h_row[pos] 
			found_next_node = True
	return path, nodes

def serialize_correlation_matrix(
		a: np.ndarray,
		serialization_fname: str,
		round_index: int,
		correlation_matrix_serialization_directory: str,
) -> None:
	numpy_fname = (
		f'{correlation_matrix_serialization_directory}/'
		+ f'{serialization_fname}_correlation_matrix'
		+ f'_{round_index}.npy'
	)
	np.save(
		numpy_fname,
		a
	)

def serialize_suboptimal_paths(
		src: int,
		sink: int,
		serialization_fname: str,
		ssp: np.ndarray,
		nodes: Nodes,
		s: Set,		
		round_index: int,
		suboptimal_paths_serialization_directory: str,
) -> None:
	d = {i[-1] : i[:-1] for i in s}
	d[ssp[-1]] = ssp[:-1]
	path_index = 0
	suboptimal_paths = SuboptimalPaths()
	for k in sorted(d):
		path = Path()
		path.length = k
		path.edges = [] 
		path_nodes = set([])
		for path_edge in ordered_paths(d[k], src):
			node1_index, node2_index = path_edge
			path_nodes.add(node1_index)
			path_nodes.add(node2_index)
			edge = Edge()	
			edge.node1 = nodes[node1_index]
			edge.node2 = nodes[node2_index]
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
	xml_fname = (
		f'{suboptimal_paths_serialization_directory}/'
		+ f'{serialization_fname}_suboptimal_paths'
		+ f'_{round_index}.xml'
	)
	suboptimal_paths.serialize(xml_fname)

def explore_paths(
		src: int, 
		sink: int, 
		a: np.ndarray, 
		nodes: List, 
		n: int, 
		cutoff: Union[float, None], 
		threads_per_block: int,
		serialization_fname: str,
		serialization_frequency: int,
		nodes_obj: Nodes,
		ssp: np.ndarray,
		round_index: int,
		correlation_matrix_serialization_directory: str,
		suboptimal_paths_serialization_directory: str,
		max_num_paths: int,
		rule: Rule,
) -> Set:
	if serialization_frequency:
		serialize_correlation_matrix(
			a,	
			serialization_fname,
			round_index,
			correlation_matrix_serialization_directory,
		)
	h = np.array(hedetniemi_distance(
		a, 
		n, 
		threads_per_block,
		cutoff,
	))
	q = deque([])
	s = set([])
	n_p = set([])
	for i in nodes:
		q.append(i)
		n_p.add(i)
	start = time.time()
	while q:
		if len(s) > max_num_paths:
			break
		for i in nodes:
			n_p.add(i)
		i, j = rule.pop(q)
		a[i][j] = np.inf
		a[j][i] = np.inf
		h = np.array(hedetniemi_distance(
			a, 
			n, 
			threads_per_block, 
			cutoff
		))
		if get_ssp(src, sink, h, a) is not None:
			path, nodes = get_ssp(src, sink, h, a)
		else:
			break
		path.append(h[src][sink])
		prev_s_size = len(s)
		s.add(tuple(path))
		new_s_size = len(s)
		if new_s_size != prev_s_size:
			if serialization_frequency:
				if (time.time() - start) > serialization_frequency:
					serialize_correlation_matrix(
						a,
						serialization_fname,
						round_index,
						correlation_matrix_serialization_directory,
					)
					serialize_suboptimal_paths(
						src,
						sink,
						serialization_fname,
						ssp,
						nodes_obj,
						s,	
						round_index,
						suboptimal_paths_serialization_directory,
					)	
					start = time.time()
		nodes = [
			(nodes[i], nodes[i+1]) 
			for i in range(len(nodes)-1)
		]
		for i in nodes:
			if i not in n_p:
				rule.append(q, i)
		if not q:
			break
	return s

def get_suboptimal_paths(
		calculation_name: str,
		input_directory: str, 
		correlation_matrix_fname: str,
		nodes_fname: str,
		src: int, 
		sink: int,
		suboptimal_paths_fname: str, 
		cutoff: Union[float, None],
		threads_per_block: int,
		serialization_fname: str,
		serialization_frequency: int,
		correlation_matrix_serialization_directory: str,
		suboptimal_paths_serialization_directory: str,
		simulation_rounds: int,
		gpu_index: int,
		max_num_paths: int,
		rules: Dict,
		apsp_matrix_fname: str,
) -> None:
	cuda.select_device(gpu_index)
	suboptimal_paths_dict = {}	
	nodes_obj = Nodes()
	nodes_obj.deserialize(nodes_fname)
	for simulation_round in simulation_rounds:
		a = np.array(np.load(
			correlation_matrix_fname
		))
		n = len(a)
		h = np.array(hedetniemi_distance(
			a, 
			n, 
			threads_per_block, 
			np.inf,
		))
		if get_ssp(src, sink, h, a) is None:
			raise Exception(
				"Sink node is unreachable from source node.".upper() + '\n'
				+ "Either perform the suboptimal path calculation" + '\n'
				+ "using the correlation matrix without the contact map" + '\n'
				+ "applied, or rerun the correlation matrix calculation with" + '\n'
				+ "a larger cutoff distance."
			)

		if not os.path.exists(
			apsp_matrix_fname
		):
			np.save(
				apsp_matrix_fname,
				h,
			)
		path, nodes = get_ssp(src, sink, h, a)
		ssp = path
		ssp.append(h[src][sink])
		if not cutoff:
			cutoff = ssp[-1] * np.float64(1.2) 
		nodes = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
		paths = list(explore_paths(
			src,
			sink,
			a,
			nodes,
			n,
			cutoff,
			threads_per_block,
			serialization_fname,
			serialization_frequency,
			nodes_obj,
			ssp,
			simulation_round,
			correlation_matrix_serialization_directory,
			suboptimal_paths_serialization_directory,
			max_num_paths,
			rules[simulation_round],
		))
		for path in paths:
			suboptimal_paths_dict[path[-1]] = path[:-1]	
	suboptimal_paths_dict[ssp[-1]] = ssp[:-1]
	path_index = 0
	suboptimal_paths = SuboptimalPaths()
	for path_length in sorted(suboptimal_paths_dict):
		path = Path()
		path.length = path_length
		path.edges = [] 
		path_nodes = set([])
		for path_edge in ordered_paths(
			suboptimal_paths_dict[path_length], src
		):
			node1_index, node2_index = path_edge
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
	suboptimal_paths.serialize(
		suboptimal_paths_fname
	)
