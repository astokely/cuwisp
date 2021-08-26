from typing import Tuple
from scipy import interpolate
import numpy as np
from cuwisp import calculate_correlation_matrix as correlation_matrix 
from cuwisp import calculate_suboptimal_paths 
from cuwisp.paths import SubOptimalPaths  
from colour import Color
import graph_correlation_matrix as graph

correlation_matrix(
	"example_output",
	10.0,
	"/home/astokely/spike/spike_example.pdb",
	cuda_parameters = (256, 100, 1024, 1000),
	#temp_file_directory = 'tmp',
)

calculate_suboptimal_paths(
	"example_output", 
	80, 
	100, 
	threads_per_block=1024, 
	cutoff=3.1,
	use_contact_map_correlation_matrix = False,
)

suboptimal_paths = SubOptimalPaths()
suboptimal_paths.deserialize("example_output/suboptimal_paths.xml")
