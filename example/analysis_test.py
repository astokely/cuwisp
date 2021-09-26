from cuwisp.analysis import sp_splines
from cuwisp.analysis import sp_average_manhattan_distances
from cuwisp.analysis import sp_frechet_distances
from cuwisp.analysis import sp_frechet_distance_matrices
from cuwisp.paths import SuboptimalPaths
import matplotlib.pyplot as plt



sp = SuboptimalPaths()
sp.deserialize('example_output/suboptimal_paths.xml')
splines =  sp_splines(sp, 17)
fdm = sp_frechet_distance_matrices(
	splines,
	num_partitions=5,
) 
print(fdm)
