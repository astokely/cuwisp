from typing import Tuple
from scipy import interpolate
import numpy as np
from cuwisp import calculate_correlation_matrix as correlation_matrix 
from cuwisp import calculate_suboptimal_paths 
from cuwisp.paths import SubOptimalPaths  
from colour import Color
import graph_correlation_matrix as graph
import abserdes.Serializer as serializer

class Visualize(serializer):

	def __init__(self):

class Material(serializer):

	def __init__(self):
		self.name = ''
		self.ambient = 0.0
		self.specular = 0.0
		self.diffuse = 0.79 
		self.shininess = 0.53
		self.mirror = 0.0
		self.opacity = 1.0

class VmdTcl(serializer)

def visualize(
		color_gradient: Tuple,
) -> None:
	correlation_matrix(
		"example_output",
		10.0,
		"spike_example.pdb",
		cuda_parameters = (256, 100, 1024, 1000),
	)

	calculate_suboptimal_paths(
		"example_output", 
		80, 
		100, 
		threads_per_block=1024, 
		cutoff=3.1
	)

	suboptimal_paths = SubOptimalPaths()
	suboptimal_paths.deserialize("example_output/suboptimal_paths.xml")

	with open("paths.tcl", "w") as tcl:
		color1, color2 = color_gradient
		color_index = 0
		color1 = Color(str(color1.lower()))
		colors = list(color1.range_to(Color(color2.lower()), suboptimal_paths.num_paths))
		num_radii = suboptimal_paths.num_paths
		radii = np.linspace(0.2, 0.05, num_radii)
		tcl.write('set avail_materials [material list]\n')
		tcl.write(
			'if {[lsearch -exact $avail_materials "node"] >= 0} {'
				+ 'material delete node'
			+'}\n'
		)
		tcl.write('material add node\n')
		tcl.write('material change ambient node 0.0\n') 
		tcl.write('material change specular node 0.0\n') 
		tcl.write('material change diffuse node 0.79\n')
		tcl.write('material change shininess node 0.53\n') 
		tcl.write('material change mirror node 0.0\n')
		tcl.write('material change opacity node 1.0\n') 
		tcl.write('set color_start [colorinfo num] \n') 

		tcl.write('set n [expr [molinfo [molinfo top] get numreps] - 1] \n') 
		tcl.write(
			'while {$n >= 0} {'
				+ ' mol delrep $n [molinfo top];' 
				+ ' set r [expr $n - 1];' 
				+ ' set n $r'
			+'}\n'
		)
		tcl.write('set molid [molinfo top] \n') 
		tcl.write('mol color name \n') 
		tcl.write('mol representation lines \n') 
		tcl.write('mol addrep $molid \n')
		tcl.write('mol modselect 0 $molid all \n')
		tcl.write('mol representation Licorice \n') 

		for path in suboptimal_paths:
			if path.num_edges < 3:
				degree = path.num_edges
			else:
				degree = 3 
			x = [edge.node1.coordinates[0] for edge in path]
			y = [edge.node1.coordinates[1] for edge in path]
			z = [edge.node1.coordinates[2] for edge in path]
			x.append(path.edges[-1].node2.coordinates[0])
			y.append(path.edges[-1].node2.coordinates[1])
			z.append(path.edges[-1].node2.coordinates[2])
			x = np.array(x)
			y = np.array(y)
			z = np.array(z)
			tck, _ = interpolate.splprep([x, y, z], s=0.0, k=degree)

			# now interpolate
			unew = np.arange(0, 1.01, 0.001)

			out = interpolate.splev(unew, tck)
			red, green, blue = colors[color_index].rgb	
			color_index += 1
			tcl.write('set color_start [colorinfo num] \n') 
			tcl.write('set red ' + str(red) + '\n') 
			tcl.write('set green ' + str(green) + '\n') 
			tcl.write('set blue ' + str(blue) + '\n') 
			tcl.write('color change rgb [expr ' + str(color_index)  + ' + $color_start] $red $green $blue \n')
			tcl.write('graphics top color [expr ' + str(color_index)  + ' + $color_start] \n')
			for i in range(len(out[0])-1):
				c1x = out[0][i] 
				c1y = out[1][i] 
				c1z = out[2][i] 
				c2x = out[0][i+1] 
				c2y = out[1][i+1] 
				c2z = out[2][i+1] 
				tcl.write('set c1 {' + str(c1x) + ' ' + str(c1y) + ' ' + str(c1z) + '}\n') 
				tcl.write('set c2 {' + str(c2x) + ' ' + str(c2y) + ' ' + str(c2z) + '}\n') 
				tcl.write('draw cylinder $c1 $c2 radius  ' + str(radii[path.index]) + ' resolution 100 filled 0 \n') 
			for edge in path:
				x, y, z = edge.node1.coordinates
				tcl.write('set c1 {' + str(x) + ' ' + str(y) + ' ' + str(z) + '}\n') 
				x, y, z = edge.node2.coordinates
				tcl.write('set c2 {' + str(x) + ' ' + str(y) + ' ' + str(z) + '}\n') 
				if edge.node1.index == path.src:
					tcl.write('draw color blue\n') 
					tcl.write('draw sphere $c1 radius 1 resolution 250\n') 
				else:
					tcl.write('draw color silver\n') 
					tcl.write('draw sphere $c1 radius .5 resolution 250\n') 
				if edge.node2.index == path.sink:
					tcl.write('draw color red\n') 
					tcl.write('draw sphere $c2 radius 1 resolution 250\n') 
				else:
					tcl.write('draw color silver\n') 
					tcl.write('draw sphere $c2 radius .5 resolution 250\n') 
				s1 = ''
				for index in edge.node1.atom_indices:
					s1 += str(index) + ' '
				s2 = ''
				for index in edge.node2.atom_indices:
					s2 += str(index) + ' '
				tcl.write('set molid [molinfo top]\n')
				tcl.write('mol representation $molid Licorice 0.2\n') 
				tcl.write('mol color name \n') 
				tcl.write('set s1 [atomselect top "index ' + s1 + '"]\n') 
				tcl.write('set s2 [atomselect top "index ' + s2 + '"]\n') 
				tcl.write('set i1 [$s1 list]\n') 
				tcl.write('set i2 [$s2 list]\n') 
				tcl.write('mol addrep $molid\n') 
				tcl.write('set repid [expr [molinfo $molid get numreps] - 1]\n') 
				tcl.write('mol modselect $repid $molid index $i1\n') 
				tcl.write('mol modmaterial $repid $molid node\n') 
				tcl.write('set repid [expr [molinfo $molid get numreps] - 1]\n') 
				tcl.write('mol modselect $repid $molid index $i2\n') 
				tcl.write('mol modmaterial $repid $molid node\n') 

	tcl.close()

visualize(("red", "green"))
