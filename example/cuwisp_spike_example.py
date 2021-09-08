import numpy as np
from cuwisp import calculate_suboptimal_paths 
from cuwisp import calculate_correlation_matrix 
from cuwisp.paths import SuboptimalPaths  
import cuwisp.vmdtcl as vmdtcl  
import cuwisp.visualize as visualize  
from colour import Color
calculate_correlation_matrix(
	"example_output",
	4.5,
	'/home/astokely/Downloads/wisp/example_commandline/trajectory_20_frames.pdb',
	cuda_parameters = (64, 10, 256, 100),
)
calculate_suboptimal_paths(
	"example_output", 
	9, 
	10,
	threads_per_block=1024, 
	cutoff=3.2,
)

suboptimal_paths = SuboptimalPaths()
suboptimal_paths.deserialize("example_output/suboptimal_paths.xml")
