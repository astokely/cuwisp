import os
import cuwisp.cuwispio as cuwispio
from cuwisp import CorrelationMatrixCalculation
from cuwisp import SuboptimalPathsCalculation

expanse_dir = '/home/astokely//expanse/lustre/scratch/astokely' \
              '/temp_project/p53'
cuio = cuwispio.IO(
    calc_name='p53',
    root_directory=expanse_dir,
    trajectory_fname=f'{os.getcwd()}/p53.pdb'
)
cuio.create_filetree()
correlation_matrix_calc = CorrelationMatrixCalculation(
    cuwisp_io=cuio,
    contact_map_distance_cutoff=4.5,
)
print(cuio)
correlation_matrix_calc.calculate_correlation_matrix()
correlation_matrix_calc.serialize(
    xml_filename='cm.xml'
)
os.system('mv /home/astokely//expanse/lustre/scratch/astokely'
          '/temp_project/p53 .' )
cuio.update(
    root_directory=f'{os.getcwd()}/p53'
)
print('\n', cuio)
suboptimal_paths_calc = SuboptimalPathsCalculation(
    cuwisp_io=cuio,
    src=1433,
    sink=1503,
    serialization_frequency=15,
)
suboptimal_paths_calc.serialize('sp.xml')
suboptimal_paths_calc.calculate_suboptimal_paths()

print('\n', cuio)
