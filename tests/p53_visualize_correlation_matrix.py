from cuwisp.vmdtcl import save_tcl
from cuwisp.visualize import visualize_correlation_matrix
from cuwisp.nodes import Nodes
from cuwisp.visualize import VisualizeCorrelationMatrix

calc = 'p53'
nodes = Nodes()
nodes_xml = f'{calc}/{calc}_nodes.xml'
nodes.deserialize(nodes_xml)
tcl = ''

params = VisualizeCorrelationMatrix(
    correlation_matrix_fname=f'{calc}/'
                       f'{calc}_contact_map_correlation_matrix'
                             f'.npy',
    nodes=nodes,
    reference_node_index=175,
    color=('white', 'purple', 'yellow')
)

tcl = visualize_correlation_matrix(
    params,
    tcl=tcl,
)
save_tcl(tcl, f'{calc}/{calc}_correlation_matrix.tcl')
