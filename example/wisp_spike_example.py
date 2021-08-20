from wisp import calculate_correlation_matrix as correlation_matrix 
from wisp import calculate_suboptimal_paths 
from wisp.paths import SubOptimalPaths  
from graph_correlation_matrix import graph
correlation_matrix(
	"example_output",
	10.0,
	"spike_example.pdb",
)

calculate_suboptimal_paths("example_output", 80, 110)

suboptimal_paths = SubOptimalPaths()
suboptimal_paths.deserialize("example_output/suboptimal_paths.xml")

'''
for path in suboptimal_paths:
	for edge in path:
		for node in edge:
			print(node.atom_indices)
'''

'''
for path in suboptimal_paths:
	for edge in path:
		for node in edge:
			print(node.identifier)
'''

for path in suboptimal_paths:
	print(path, path.length)

#print(suboptimal_paths.src)

#print(suboptimal_paths.sink)

#print(suboptimal_paths.num_paths)

graph()
