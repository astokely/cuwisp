from cuwisp.nodes import Nodes

nodes = Nodes()
nodes.deserialize('example_output/nodes.xml')
for node in nodes.find_nodes(segment_id='MEAO'):
	print(node)
print('')
for node in nodes.find_nodes(chain_index=2):
	print(node)

print('')
for node in nodes.find_nodes(chain_index=2, resname='BGLN'):
	print(node)

print('')
print(nodes.get_node_from_atom_index(100))
