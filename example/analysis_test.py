from cuwisp.analysis import sp_splines
from cuwisp.analysis import sp_manhattan_distance_matrix as mdm
from cuwisp.analysis import sp_frechet_distance_matrix
from cuwisp.analysis import sort_distances as sort
from cuwisp.paths import SuboptimalPaths
import matplotlib.pyplot as plt



sp = SuboptimalPaths()
sp.deserialize('example_output/suboptimal_paths.xml')
splines =  sp_splines(sp, 17)
'''
fdm = sp_frechet_distance_matrix(
	0,
	splines,
	num_partitions=5,
) 

for j in fdm:
	print(sort(j).keys())

fdm = sp_frechet_distance_matrix(
	0,
	splines,
) 
for j in fdm:
	print(sort(j).keys())

'''
for i in range(len(sp.paths)):
	md = mdm(
		i,
		splines,
		num_partitions=10,
	) 
	fd = sp_frechet_distance_matrix(
		i,
		splines,
		num_partitions=10,
	) 
	for j in md:
		print(sort(j))
	for j in fd:
		print(sort(j))
