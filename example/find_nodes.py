from cuwisp.nodes import Nodes

nodes = Nodes()
nodes.deserialize('example_output/nodes.xml')
#find all nodes that have resname LEU
for node in nodes.find_nodes(resid=1):
	print(node)

