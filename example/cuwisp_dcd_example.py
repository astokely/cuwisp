import numpy as np
from cuwisp import calculate_suboptimal_paths 
from cuwisp import calculate_correlation_matrix 
from cuwisp.paths import SuboptimalPaths  
import time

start = time.time()
calculate_correlation_matrix(
	"example_output", #output directory
	4.5, #contact map cutoff limit
	'test.dcd', #input pdb
	topology_filename='top.pdb',
	cuda_parameters = (64, 10, 256, 100), #Probably don't change these
	num_multiprocessing_processes = 50,
)
calculate_suboptimal_paths(
	"example_output", #same as above 
	9, #src node 
	10, #sink node
	threads_per_block=1024, #You shouldn't have to change this, but if you get a cuda error change it to 256 
	cutoff=3.2, #Cutoff correlation path length...you may have to decrease/increase this. A lower value will make the calculation faster but will not output as many paths
)
print(time.time()-start)
