import numpy as np
from cuwisp import calculate_suboptimal_paths 
from cuwisp import calculate_correlation_matrix 
from cuwisp.paths import SuboptimalPaths  
import cuwisp.vmdtcl as vmdtcl  
import cuwisp.visualize as visualize  
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
	use_contact_map_correlation_matrix=False,
	serialization_xml_filename='spike.xml',
	serialization_frequency=0.05, #In seconds
	simulation_round_index=0,
	correlation_matrix_serialization_path='correlation_matrices',
	suboptimal_paths_serialization_path='suboptimal_paths',
)

suboptimal_paths = SuboptimalPaths()
suboptimal_paths.deserialize("example_output/suboptimal_paths.xml")
