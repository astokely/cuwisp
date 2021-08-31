from typing import Tuple
from scipy import interpolate
import numpy as np
from cuwisp import calculate_suboptimal_paths 
from cuwisp import calculate_correlation_matrix 
from cuwisp.paths import SuboptimalPaths  
from colour import Color
calculate_correlation_matrix(
	"example_output",
	10.0,
	'spike_example.pdb',
	cuda_parameters = (256, 100, 1024, 1000),
)
calculate_suboptimal_paths(
	"example_output", 
	80, 
	100,
	threads_per_block=1024, 
	#use_contact_map_correlation_matrix=False,
	cutoff=4.5,
)

suboptimal_paths = SuboptimalPaths()
suboptimal_paths.deserialize("example_output/suboptimal_paths.xml")
