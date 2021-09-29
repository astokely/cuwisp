from cuwisp.analysis import sp_splines
from cuwisp.analysis import sp_frechet_distance_matrix
from cuwisp.analysis import sort_distances as sort
from cuwisp.analysis import sp_distance_matrix 
from cuwisp.analysis import sp_distance_matrix 
from cuwisp.analysis import load_splines 
from cuwisp.paths import SuboptimalPaths
from cuwisp.analysis import Analysis 
from cuwisp.analysis import FrechetDistanceMatrix 
from cuwisp.analysis import DistanceMatrix 
import matplotlib.pyplot as plt
import numpy as np

sp = SuboptimalPaths()
sp.deserialize('example_output/suboptimal_paths.xml')
analysis = Analysis()
splines =  sp_splines(sp, 17, output_directory='splines', spline_input_points_incr=0.01)
fdm = np.save('fdm.npy', sp_frechet_distance_matrix(5, splines, 1)) 
dm = np.save('dm.npy', sp_distance_matrix(5, splines, 1)) 

analysis.frechet_distance_matrices[5] = FrechetDistanceMatrix('fdm', 5, 1)
analysis.distance_matrices[5] = DistanceMatrix('dm', 5, 1)

analysis.serialize('analysis.xml')
analysis = Analysis()

analysis.deserialize('analysis.xml')
for m in analysis.frechet_distance_matrices.values():
	print(m.ordered(0).values())

for m in analysis.distance_matrices.values():
	print(m.ordered(0).values())

