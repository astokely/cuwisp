from cuwisp.vmdtcl import save_tcl
from cuwisp.visualize import VisualizeCorrelationMatrix, visualize_correlation_matrix

params = VisualizeCorrelationMatrix(
	node_index=9, 
	nodes_xml_filename='example_output/nodes.xml', 
	correlation_matrix_filename='correlation_matrix.txt', 
	color=('red', 'green'),
	node_color='blue',
	node_sphere_radius=2.0,
)
tcl = ''
tcl = visualize_correlation_matrix(params,tcl=tcl)
save_tcl(tcl, 'node9_correlations.tcl')

