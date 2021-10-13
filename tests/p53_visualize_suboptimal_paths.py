from cuwisp.paths import SuboptimalPaths
from cuwisp.visualize import draw_suboptimal_paths
from cuwisp.visualize import VisualizeSuboptimalPaths
from cuwisp.vmdtcl import save_tcl

calc = 'p53'
suboptimal_paths = SuboptimalPaths()
fname = ('/home/andy/PycharmProjects/cuwisp/tests/p53' 
         f'/P53SuboptimalPathsSerialization'
         f'/SerializedSuboptimalPaths_0_1_2_3_4/backup'
         f'/SerializedP53SuboptimalPaths4.xml')

suboptimal_paths.deserialize(fname)
for path in suboptimal_paths.paths:
    path.serialized_splines = [
        f'splines/frame0/spline{path.index}.xml']
params = VisualizeSuboptimalPaths(
    suboptimal_paths=suboptimal_paths,
    radii=(0.3, 0.3),
    color=('green', 'red'),
    src_node_sphere={
        'color': 'blue',
        'radius': 2.0
    },
    sink_node_sphere={
        'color': 'red',
        'radius': 2.0
    },
    path_indices=list(range(len(suboptimal_paths.paths)))
)

tcl = ''
tcl = draw_suboptimal_paths(
    parameters=params,
    tcl=tcl,
)

save_tcl(
    tcl=tcl,
    tcl_filename=f'{calc}.tcl'
)
