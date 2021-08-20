from cuwisp import calculate_correlation_matrix as correlation_matrix 
from cuwisp import calculate_suboptimal_paths 
from cuwisp.paths import SubOptimalPaths  
import graph_correlation_matrix as graph
correlation_matrix(
	"example_output",
	10.0,
	"spike_example.pdb",
	cuda_parameters = (1024, 10, 1024, 1000)
)

calculate_suboptimal_paths("example_output", 80, 110, threads_per_block=1024)

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

graph.plot_heatmap()
graph.plot_3D_surface()
