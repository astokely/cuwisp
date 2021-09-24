from cuwisp.visualize import VisualizeSuboptimalPaths
from cuwisp.visualize import VmdRepresentation
from cuwisp.visualize import draw_suboptimal_paths 
from cuwisp.paths import SuboptimalPaths
from cuwisp.vmdtcl import save_tcl
from cuwisp.vmdtcl import Licorice 
from cuwisp.nodes import Nodes

tcl = ''
suboptimal_paths = SuboptimalPaths()
suboptimal_paths.deserialize('example_output/suboptimal_paths.xml')
nodes = Nodes()
nodes.deserialize('example_output/nodes.xml')

node_reps = {} 

for node in nodes:
	node_reps[f'node{node.index}'] = VmdRepresentation(
		selection=('index', node.atom_indices),
		style=Licorice(),
	)
		
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
	node_atoms_representations = node_reps,
	frame = 17,
)
tcl = draw_suboptimal_paths(
	visualize_suboptimal_paths,
	tcl=tcl
)

save_tcl(tcl, 'suboptimal_paths.tcl')
