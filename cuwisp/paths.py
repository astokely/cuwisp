import mdtraj as md
from collections import deque
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from numba import cuda, njit, float32
import numpy as np
import math
import heapq
from typing import Any, Tuple, Optional, List, Union, Set
from math import ceil
from math import floor
from abserdes import Serializer as serializer
from .nodes import Nodes, Node

class SubOptimalPaths(serializer):

	def __init__(self) -> None:
		self.paths = []
		self.src = None
		self.sink = None
		self.num_paths = None
		return

	def __iter__(self):
		for path in self.paths:
			yield path

class Path(serializer):

	def __init__(self) -> None:
		self.index = None
		self.src = None
		self.sink = None
		self.edges = [] 
		self.length = None

	def __repr__(self):
		return str(self.edges)

	def __iter__(self):
		for edge in self.edges:
			yield edge 

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
		nodes = [self.node1, self.node1]
		for node in nodes:
			yield node 

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

def deque_append_middle(
		dq: deque, 
		val: Any,
) -> Any:
	middle_index = floor(len(dq)/2)
	dq.insert(middle_index, val)

def deque_pop_middle(
		dq: deque,
) -> Any:
	middle_index = floor(len(dq)/2)
	middle_val = dq[middle_index]
	del dq[middle_index]
	return middle_val



def cuda_grid_and_block_dims(
		num_nodes: int, 
		threads_per_block: int,
) -> Tuple:
	dimBlock = (threads_per_block, threads_per_block)
	dimGrid = (
		math.ceil(num_nodes / threads_per_block), 
		math.ceil(num_nodes / threads_per_block)
	)
	return dimGrid, dimBlock

@cuda.jit
def init_matrix(
		a,
		b,
		h,
		num_nodes,
):
	i, j = cuda.grid(2)
	if i < num_nodes and j < num_nodes:
		h[i,j] = np.inf
		b[i,j] = a[i,j]

@cuda.jit
def all_pair_hedetniemit(
		a, 
		b, 
		h, 
		num_nodes,
		found_ssp
	):
	i, j = cuda.grid(2)
	if i < num_nodes  and j < num_nodes:
		hedetniemit_sum = np.inf
		for a_row_b_col_index in range(num_nodes):
			hedetniemit_sum = min(
				hedetniemit_sum, 
				b[i, a_row_b_col_index] + a[a_row_b_col_index, j]
			)
		h[i,j] = hedetniemit_sum
		if b[i,j] != h[i,j]:	  
			b[i,j] = h[i,j]
			found_ssp[0] = False

def hedetniemi_distance(
		a: Any,
		num_nodes: int,
		max_edges_in_ssp: int,
		threads_per_block: int,
) -> Any:
	threads_per_block = int(np.sqrt(threads_per_block))
	n = num_nodes
	a_device = cuda.to_device(a)
	b_device = cuda.device_array(shape=(n,n))
	h_device = cuda.device_array(shape=(n,n))
	dimGrid, dimBlock = cuda_grid_and_block_dims(n, threads_per_block)
	init_matrix[dimGrid, dimBlock](
		a_device, 
		b_device, 
		h_device, 
		num_nodes
	)
	for i in range(max_edges_in_ssp):
		found_ssp = cuda.to_device([True])
		dimGrid, dimBlock = cuda_grid_and_block_dims(n, threads_per_block)
		all_pair_hedetniemit[dimGrid, dimBlock](
			a_device, 
			b_device, 
			h_device, 
			num_nodes, 
			found_ssp
		)
		if found_ssp[0] == True:
			break
	h_host = h_device.copy_to_host()
	return h_host 


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
	heapq.heappush(nodes, pos)
	while pos != src:
		closest = {} 
		found_next_node = False
		for i in range(len(h_row)):
			dist = h_row[i] + a_col[i]
			if i != pos:
				if p == dist == np.inf:
					return 
				closest[abs(dist-p)] = i 
			if dist == p and i != pos:
				prev_pos = pos 
				pos = i
				a_col = a[pos,:]
				heapq.heappush(path, (pos, prev_pos))
				heapq.heappush(nodes, pos)
				p = h_row[pos] 
				found_next_node = True
				break
		if not found_next_node:
			minn = min(closest.keys())
			prev_pos = pos 
			pos = closest[minn]
			a_col = a[pos,:]
			heapq.heappush(path, (pos, prev_pos))
			heapq.heappush(nodes, pos)
			p = h_row[pos] 
			found_next_node = True
	return path, nodes

def explore_paths(
		src: int, 
		sink: int, 
		a: np.ndarray, 
		nodes: List, 
		n: int, 
		pop: int,
		cutoff: Union[float, None], 
		threads_per_block: int,
) -> Set:
	h = np.array(hedetniemi_distance(a, n, n, threads_per_block))
	q = deque([])
	s = set([])
	n_p = set([])
	for i in nodes:
		q.append(i)
		n_p.add(i)
	while q:
		for i in nodes:
			n_p.add(i)
		if pop == 0:
			i, j = q.pop()
		elif pop == 1:
			i, j = deque_pop_middle(q)
		elif pop == 2:
			i, j = q.popleft()
		elif pop == 3:
			i, j = q.popleft()
		elif pop == 4:
			i, j = q.pop()
		a[i][j] = np.inf
		a[j][i] = np.inf
		h = np.array(hedetniemi_distance(a, n, n, threads_per_block))
		if get_ssp(src, sink, h, a) is not None:
			path, nodes = get_ssp(src, sink, h, a)
		else:
			break
		path.append(h[src][sink])
		prev_s_size = len(s)
		s.add(tuple(path))
		new_s_size = len(s)
		nodes = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
		for i in nodes:
			if i not in n_p:
				if pop == 0:
					q.append(i)
				elif pop == 1:
					deque_append_middle(q, i)
				elif pop == 2:
					q.append(i)
				elif pop == 3:
					deque_append_middle(q, i)
				elif pop == 4:
					deque_append_middle(q, i)
		if cutoff is not None:
			if h[src][sink] >= cutoff:
				break
		if not q:
			break
	return s

def get_suboptimal_paths(
		input_files_path: str, 
		correlation_matrix_file: str,
		nodes_xml_file: str,
		src: int, 
		sink: int,
		suboptimal_paths_xml: str, 
		cutoff: Union[float, None],
		threads_per_block: int,
):
	suboptimal_paths = SubOptimalPaths()
	a = np.array(np.loadtxt(input_files_path + "/" + correlation_matrix_file))
	n = len(a)
	h = np.array(hedetniemi_distance(a, n, n, threads_per_block))
	if get_ssp(src, sink, h, a) is None:
		raise Exception(
			"Sink node is unreachable from source node.".upper() + '\n'
			+ "Either perform the suboptimal path calculation" + '\n'
			+ "using the correlation matrix without the contact map" + '\n'
			+ "applied, or rerun the correlation matrix calculation with" + '\n'
		    + "a larger cutoff distance."
		)
	path, nodes = get_ssp(src, sink, h, a)
	ssp = path
	ssp.append(h[src][sink])
	nodes = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
	paths1 = list(explore_paths(src, sink, a, nodes, n, 0, cutoff, threads_per_block))
	d1 = {i[-1] : i[:-1] for i in paths1}

	a = np.array(np.loadtxt(input_files_path + "/" + correlation_matrix_file))
	n = len(a)
	h = np.array(hedetniemi_distance(a, n, n, threads_per_block))
	path, nodes = get_ssp(src, sink, h, a)
	ssp = path
	ssp.append(h[src][sink])
	nodes = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
	paths2 = list(explore_paths(src, sink, a, nodes, n, 1, cutoff, threads_per_block))
	d2 = {i[-1] : i[:-1] for i in paths2}

	a = np.array(np.loadtxt(input_files_path + "/" + correlation_matrix_file))
	n = len(a)
	h = np.array(hedetniemi_distance(a, n, n, threads_per_block))
	path, nodes = get_ssp(src, sink, h, a)
	ssp = path
	ssp.append(h[src][sink])
	nodes = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
	paths3 = list(explore_paths(src, sink, a, nodes, n, 2, cutoff, threads_per_block))
	d3 = {i[-1] : i[:-1] for i in paths3}

	a = np.array(np.loadtxt(input_files_path + "/" + correlation_matrix_file))
	n = len(a)
	h = np.array(hedetniemi_distance(a, n, n, threads_per_block))
	path, nodes = get_ssp(src, sink, h, a)
	ssp = path
	ssp.append(h[src][sink])
	nodes = [(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)]
	paths4 = list(explore_paths(src, sink, a, nodes, n, 3, cutoff, threads_per_block))
	d4 = {i[-1] : i[:-1] for i in paths4}

	d = {}
	d.update(d1)
	d.update(d2)
	d.update(d3)
	d.update(d4)
	d[ssp[-1]] = ssp[:-1]
	nodes = Nodes()
	nodes.deserialize(input_files_path + "/" + nodes_xml_file)
	path_index = 0
	for k in sorted(d):
		path = Path()
		path.length = k
		path.edges = [] 
		for path_edge in ordered_paths(d[k], src):
			node1_index, node2_index = path_edge
			edge = Edge()	
			edge.node1 = nodes[node1_index]
			edge.node2 = nodes[node2_index]
			path.edges.append(edge)
		path.src = src
		path.sink = sink 
		path.index = path_index
		suboptimal_paths.paths.append(path)
		path_index += 1
	suboptimal_paths.src = src
	suboptimal_paths.sink = sink 
	suboptimal_paths.num_paths = len(suboptimal_paths.paths) 
	suboptimal_paths.serialize(input_files_path + "/" + suboptimal_paths_xml)

