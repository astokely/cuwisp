from abserdes import Serializer as serializer

class Node(serializer):

	def __init__(self):
		self.index = 0
		self.atom_indices = []
		self.identifier = ''
		self.coordinates = None

	def __repr__(self):
		return str(self.index)

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
		if index <= len(self.nodes):
			return self.nodes[index]
		return

	def get_node_from_atom_index(
			self, 
			index: int,
	) -> Node:
		for node in self.nodes:
			if index in node.atom_indices:
				return node
