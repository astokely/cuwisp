import numpy as np
from .cuComReduction import centerOfMassReduction


def calc_com(all_indices, all_coords, all_masses, num_blocks = 10):
	indices = all_indices[0]
	group_sizes = [len(group) for group in indices]
	largest_group = max(group_sizes)
	masses = all_masses[0]
	coordinates = all_coords
	m = []
	c = []
	for group in indices:
		gm = np.zeros(largest_group)
		i = 0
		for index in group:
			gm[i] = masses[index]
			i += 1
		m.append(gm) 

	for group in indices:
		gc = []
		for f in range(len(coordinates)):
			g = np.array([
				np.zeros(largest_group, dtype=np.float64)
				for i in range(3)
			])
			i = 0
			for index in group:
				g[0][i] = coordinates[f][index][0]
				g[1][i] = coordinates[f][index][1]
				g[2][i] = coordinates[f][index][2]
				i += 1
			gc.append(g.flatten())
		c.append(np.array(gc))
	c = np.array(c)
	m = np.array(m)
	total_masses = np.array([sum(mass_group) for mass_group in m], dtype=np.float64) 
	
	coms = centerOfMassReduction(
		c, m, total_masses, 
		256, num_blocks, largest_group, 
		len(indices), len(coordinates)
	) 
	C = []
	for i in range(len(coordinates)):
		c = np.array([np.zeros(3, dtype=np.float64) for i in range(len(coms))])
		for j in range(len(coms)):
			c[j][0] = coms[j][i][0]
			c[j][1] = coms[j][i][1]
			c[j][2] = coms[j][i][2]
		C.append(c)
	C = np.array(C)
	return C 
		


		
	




