from cuwisp.nodes import Nodes

nodes = Nodes()
#Uncomment below to see all the nodes.find_nodes arguments
#print(help(nodes.find_nodes))
nodes.deserialize('example_output/nodes.xml')

#find all nodes that have resname LEU
for node in nodes.find_nodes(resname='LEU'):
	print(node)

print('')
#find all nodes that have resname segment_id MEAO 
for node in nodes.find_nodes(segment_id='MEAO'):
	print(node)

print('')
#find all nodes that have resname segment_id MEAO and print the node index
for node in nodes.find_nodes(segment_id='MEAO'):
	print(node.index)
print('')

#find all nodes that have resname segment_id MEAO and resname POPC and print the node index
for node in nodes.find_nodes(segment_id='MEAO', resname='POPC'):
	print(node.index)

print('')

#find all nodes that are part of chain index 0 
for node in nodes.find_nodes(chain_index=0):
	print(node)
