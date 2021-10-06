from collections import deque
import numpy as np
from cuwisp import calculate_suboptimal_paths 
from cuwisp.analysis import Analysis
from cuwisp import calculate_correlation_matrix 
from cuwisp.paths import SuboptimalPaths  
from cuwisp import merge_suboptimal_paths
import time
import gc

start = time.time()
for i in range(3):
	calculate_correlation_matrix(
		f'calc{i}',
		f'calc{i}', #same as above 
		4.5, #contact map cutoff limit
		'test.pdb', #input pdb
		cuda_parameters = (64, 10, 256, 100), #Probably don't change these
		num_multiprocessing_processes = 16,
		node_coordinate_frames = [0,],
	)
	calculate_suboptimal_paths(
		f'calc{i}',
		f'calc{i}', #same as above 
		9, #src node 
		10, #sink node
		threads_per_block=1024, #You shouldn't have to change this, but if you get a cuda error change it to 256 
		#use_contact_map_correlation_matrix=False, #Probably don't change this
		simulation_rounds=[0, 1, 2, 3, 4],
		serialization_frequency=0.5,
		cutoff=3.2,
	)

	merge_suboptimal_paths(
		f'calc{i}',
		[0, 1, 2, 3, 4], 
		f'calc{i}/calc{i}_nodes.xml', 
		f'calc{i}/calc{i}_suboptimal_paths.xml'
	)
	analysis = Analysis(f'calc{i}/analysis')
	sp = SuboptimalPaths()
	sp.deserialize(f'calc{i}/calc{i}_suboptimal_paths.xml')
	analysis.prepare(sp)
	analysis.frechet_distance_matrix(num_partitions=3)
	for v in analysis.frechet_distance_matrices.values():
		print(v.delta)
print('')
print(time.time()-start)
