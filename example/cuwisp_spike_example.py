import numpy as np
from cuwisp import calculate_correlation_matrix as correlation_matrix 
from cuwisp import calculate_suboptimal_paths 
from cuwisp.paths import SubOptimalPaths  
import graph_correlation_matrix as graph
correlation_matrix(
	"example_output",
	10.0,
	"spike_example.pdb",
	cuda_parameters = (256, 100, 1024, 1000)
)

calculate_suboptimal_paths("example_output", 80, 110, threads_per_block=1024)

suboptimal_paths = SubOptimalPaths()
suboptimal_paths.deserialize("example_output/suboptimal_paths.xml")

with open("paths.tcl", "w+") as tcl:
	i = 0
	num_radii = suboptimal_paths.num_paths
	radii = np.linspace(0.2, 0.05, num_radii)
	tcl.write('set color_start [colorinfo num \n') 
	tcl.write('set r 0 \n') 
	tcl.write('set g 1 \n') 
	tcl.write('set b 0.3221 \n') 
	for path in suboptimal_paths:
		tcl.write('color change rgb [expr ' + str(i)  + ' + $color_start] $r $g $b \n')
		for edge in path:
			n1 = ''
			for atom_index in edge.node1.atom_indices:
				n1 += str(atom_index) + " "
			n1 = n1[:-1]
			n2 = ''
			for atom_index in edge.node2.atom_indices:
				n2 += str(atom_index) + " "
			n2 = n2[:-1]
			tcl.write('set n1 [atomselect top "index ' + n1 + '"]\n') 
			tcl.write('set n2 [atomselect top "index ' + n2 + '"]\n') 
			tcl.write('set c1 [measure center $n1 weight mass]\n') 
			tcl.write('set c2 [measure center $n2 weight mass]\n') 
			tcl.write('draw color blue \n') 
			tcl.write('draw sphere $c1 radius .7 resolution 6 \n') 
			tcl.write('draw sphere $c2 radius .7 resolution 6 \n') 
			tcl.write('draw color red \n') 
			tcl.write('draw cylinder $c1 $c2 radius ' + str(radii[path.index]) + ' resolution 6 filled 0\n') 
			tcl.write('draw cylinder $c1 $c2 radius  ' + str(radii[path.index]) + ' resolution 6 filled 0\n') 
		i += 1
tcl.close()
