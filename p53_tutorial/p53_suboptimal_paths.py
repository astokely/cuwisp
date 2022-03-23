import os

import cuwisp.cuwispio as cuwispio
from cuwisp import SuboptimalPathsCalculation

'''
 Make sure to unzip p53.tar.gz (tar -xvf p53.tar.gz) before 
 running the below example.
'''

# Deserialize IO object
cuio = cuwispio.IO()
cuio.deserialize('io.xml')

# Set suboptimal path calculation parameters.
suboptimal_paths_calc = SuboptimalPathsCalculation(
    cuwisp_io=cuio,
    max_num_paths=3,
    src=1402,
    sink=1503,
    cutoff=2.0,
    serialization_frequency=10
)

# Perform suboptimal path calculation.
suboptimal_paths_calc.calculate_suboptimal_paths()
