import numpy as np
from cuwisp import calculate_suboptimal_paths 
from cuwisp import calculate_correlation_matrix 
from cuwisp.paths import SuboptimalPaths  
from cuwisp import merge_suboptimal_paths_xmls
import time
import gc

start = time.time()
calculate_correlation_matrix(
	"example_output", #output directory
	4.5, #contact map cutoff limit
	#'spike_example.pdb', #input pdb
	'/home/astokely/Downloads/wisp/example_commandline/trajectory_20_frames.pdb', #input pdb
	cuda_parameters = (64, 10, 256, 100), #Probably don't change these
	node_coordinate_frames = [17],
)
calculate_suboptimal_paths(
	"example_output", #same as above 
	9, #src node 
	10, #sink node
	threads_per_block=1024, #You shouldn't have to change this, but if you get a cuda error change it to 256 
	#use_contact_map_correlation_matrix=False, #Probably don't change this
	simulation_rounds=[0, 1, 2],
	#max_edges = 2,
	serialization_filename='spike_protein',
	serialization_frequency=0.5,
	cutoff=3.2,
)
calculate_suboptimal_paths(
	"example_output", #same as above 
	9, #src node 
	10, #sink node
	threads_per_block=1024, #You shouldn't have to change this, but if you get a cuda error change it to 256 
	#use_contact_map_correlation_matrix=False, #Probably don't change this
	simulation_rounds=[3, 4],
	#max_edges = 2,
	serialization_filename='spike_protein',
	serialization_frequency=0.5,
	cutoff=3.2,
)

merge_suboptimal_paths_xmls(
	'example_output', 
	[0, 1, 2, 3, 4], 
	'example_output/nodes.xml', 
	'example_output/suboptimal_paths.xml'
)
print('')
print(time.time()-start)
