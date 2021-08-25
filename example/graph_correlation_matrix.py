from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def plot_heatmap():
	a = np.loadtxt("example_output/correlation_matrix_after_contact_map.txt")
	b = np.zeros(a.shape, dtype=np.float64)

	m, n = a.shape
	for i in range(m):
		for j in range(n):
			if a[i][j] == 0.0:
				b[i][j] = 0.0 
			else:
				b[i][j] = 1/a[i][j] 
	plt.imshow(b, cmap='viridis')
	plt.colorbar()
	plt.show()

def plot_3D_surface():
	a = np.loadtxt("example_output/correlation_matrix_after_contact_map.txt")
	b = np.zeros(a.shape, dtype=np.float64)

	m, n = a.shape
	for i in range(m):
		for j in range(n):
			if a[i][j] == 0.0:
				b[i][j] = 0.0 
			else:
				b[i][j] = 1/a[i][j] 
	c = [(i, j, b[i][j]) for i in range(m) for j in range(n)]
	x, y, z = list(zip(*c))
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	surf = ax.plot_trisurf(x, y, z)
	plt.show()
