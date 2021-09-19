from cuwisp.visualize import VisualizeSuboptimalPaths
from cuwisp.visualize import draw_suboptimal_paths 
from cuwisp.paths import SuboptimalPaths
from cuwisp.vmdtcl import save_tcl

tcl = ''
suboptimal_paths = SuboptimalPaths()
suboptimal_paths.deserialize('serialized_suboptimal_paths/spike_protein_suboptimal_paths_2_5.xml')

node_spheres = {'radius' : 1, 'color':'silver', 'resolution':250}
src_node_sphere = {'radius' : 2.0, 'color':'blue', 'resolution':250}
sink_node_sphere = {'radius' : 2.0, 'color':'purple', 'resolution':250}
visualize_suboptimal_paths = VisualizeSuboptimalPaths(
	suboptimal_paths,
	('red', 'green'),
	(0.35, .8),
	node_spheres = node_spheres,
	src_node_sphere = src_node_sphere,
	sink_node_sphere = sink_node_sphere,
)
tcl = draw_suboptimal_paths(
	visualize_suboptimal_paths,
	tcl=tcl
)

save_tcl(tcl, 'suboptimal_paths.tcl')

