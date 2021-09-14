from cuwisp.visualize import VisualizeSuboptimalPaths
from cuwisp.visualize import draw_suboptimal_paths 
from cuwisp.paths import SuboptimalPaths
from cuwisp.vmdtcl import save_tcl

tcl = ''
suboptimal_paths = SuboptimalPaths()
suboptimal_paths.deserialize('example_output/suboptimal_paths.xml') 

node_spheres = {'radius' : 0.3, 'color':'silver', 'resolution':250}
src_node_sphere = {'radius' : 0.5, 'color':'blue', 'resolution':250}
sink_node_sphere = {'radius' : 0.5, 'color':'purple', 'resolution':250}
visualize_suboptimal_paths = VisualizeSuboptimalPaths(
	suboptimal_paths,
	('red', 'green'),
	(0.03, 0.08),
	node_spheres = node_spheres,
	src_node_sphere = src_node_sphere,
	sink_node_sphere = sink_node_sphere,
)
tcl = draw_suboptimal_paths(
	visualize_suboptimal_paths,
	tcl=tcl
)

save_tcl(tcl, 'suboptimal_paths.tcl')

