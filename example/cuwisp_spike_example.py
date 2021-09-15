import numpy as np
from cuwisp import calculate_suboptimal_paths 
from cuwisp import calculate_correlation_matrix 
from cuwisp.paths import SuboptimalPaths  
import time

start = time.time()
#uncomment below for a list of input arguments/descriptions
#print(help(calculate_correlation_matrix))
calculate_correlation_matrix(
	"example_output", #output directory
	4.5, #contact map cutoff limit
	#'spike_example.pdb', #input pdb
	'test.dcd', #input pdb
	topology_filename='top.pdb',
	#'/home/astokely/spike/spike_protein.pdb', #input pdb
	#'/home/astokely/rep1.pdb',
	#'/home/astokely/Downloads/wisp/example_commandline/trajectory_20_frames.pdb',
	cuda_parameters = (64, 10, 256, 100), #Probably don't change these
	num_multiprocessing_processes = 50,
)
calculate_suboptimal_paths(
	"example_output", #same as above 
	9, #src node 
	10, #sink node
	threads_per_block=1024, #You shouldn't have to change this, but if you get a cuda error change it to 256 
	#use_contact_map_correlation_matrix=False, #Probably don't change this
	cutoff=3.2, #Cutoff correlation path length...you may have to decrease/increase this. A lower value will make the calculation faster but will not output as many paths
	#serialization_xml_filename='spike_protein.xml',
	#serialization_frequency=900,
	#correlation_matrix_serialization_directory='tmp',
	#suboptimal_paths_serialization_directory='tmp',
)
print(time.time()-start)
