from cuwisp.vmdtcl import save_tcl
from cuwisp.visualize import VisualizeCorrelationMatrix, draw_correlation_matrix

params = VisualizeCorrelationMatrix(
	node_index=100, 
	nodes_xml_filename='example_output/nodes.xml', 
	correlation_matrix_filename='example_output/correlation_matrix.txt', 
	color=('purple', 'yellow'),
	node_color='blue',
	node_sphere_radius=4.0,
)
tcl = ''
tcl = draw_correlation_matrix(params,tcl=tcl)
save_tcl(tcl, 'v.tcl')
