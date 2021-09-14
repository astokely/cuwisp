import numpy as np
from cuwisp import calculate_suboptimal_paths 
from cuwisp import calculate_correlation_matrix 
from cuwisp.paths import SuboptimalPaths  
import cuwisp.vmdtcl as vmdtcl  
import cuwisp.visualize as visualize  
from colour import Color
import time
import cProfile as c
import pstats
from pstats import SortKey

start = time.time()
calculate_correlation_matrix(
	"example_output",
	4.5,
	#'spike_example.pdb',
	'/home/astokely/spike/spike_protein.pdb',
	#'/home/astokely/Downloads/wisp/example_commandline/trajectory_20_frames.pdb',
	cuda_parameters = (64, 10, 256, 100),
)
print(time.time()-start)
'''
calculate_suboptimal_paths(
	"example_output", 
	9, 
	10,
	threads_per_block=1024, 
	use_contact_map_correlation_matrix=False,
	cutoff=3.2,
)
def calculate_sp_prof(args):
    x = calculate_suboptimal_paths(
        args[0],
        args[1],
        args[2],
        threads_per_block=args[3],
        cutoff=args[4],
    )
    return x

args = (
    "example_output",
    9,
    10,
    1024,
    3.2,
)

c.run('calculate_sp_prof(args)', 'profile')

p = pstats.Stats('profile')
p.sort_stats(SortKey.CUMULATIVE).print_stats(10)






suboptimal_paths = SuboptimalPaths()
suboptimal_paths.deserialize("example_output/suboptimal_paths.xml")
'''
