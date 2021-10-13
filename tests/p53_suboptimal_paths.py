import os
import cuwisp.cuwispio as cuwispio
from cuwisp import CorrelationMatrixCalculation
from cuwisp import SuboptimalPathsCalculation


cuio = cuwispio.IO(
    calc_name='p53',
    root_directory=f'{os.getcwd()}/p53',
    trajectory_fname=f'{os.getcwd()}/p53.pdb',
    naming_convention='pascal_case'
)
cuio.create_filetree()
correlation_matrix_calc = CorrelationMatrixCalculation(
    cuwisp_io=cuio,
    contact_map_distance_cutoff=4.5,
)
correlation_matrix_calc.calculate_correlation_matrix()
correlation_matrix_calc.serialize(
    xml_filename='cm.xml'
)
suboptimal_paths_calc = SuboptimalPathsCalculation(
    cuwisp_io=cuio,
    src=1433,
    sink=1503,
    serialization_frequency=60,
    cutoff=2.0
)
suboptimal_paths_calc.serialize('sp.xml')
suboptimal_paths_calc.calculate_suboptimal_paths()

