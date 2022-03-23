# Create IO object
from cuwisp import CorrelationMatrixCalculation
import os

from cuwisp import cuwispio

cuio = cuwispio.IO(
    calc_name='jak2_lig5',
    root_directory=f'{os.getcwd()}/jak2_lig5',
    trajectory_fname=f'{os.getcwd()}/jak2_lig.dcd',
    topology_fname=f'{os.getcwd()}/jak2_lig5_topology.pdb',
    naming_convention='camel_case'
)

# Generate calculation filetree.
cuio.create_filetree()
cuio.serialize('io.xml')

# Set correlation matrix calculation parameters
correlation_matrix_calc = CorrelationMatrixCalculation(
    cuwisp_io=cuio,
    contact_map_distance_cutoff=4.5,
)

# Perform correlation matrix calculation.
correlation_matrix_calc.calculate_correlation_matrix()
