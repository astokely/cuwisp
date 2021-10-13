from cuwisp.vmdtcl import save_tcl
from cuwisp.visualize import draw_suboptimal_paths
from cuwisp.visualize import VisualizeSuboptimalPaths
from cuwisp.paths import SuboptimalPaths

calc = 'p53'
suboptimal_paths = SuboptimalPaths()
suboptimal_paths_xml = f'{calc}/{calc}_suboptimal_paths.xml'
suboptimal_paths.deserialize(suboptimal_paths_xml)
tcl = ''
colors = [
    'blue',
    'red',
    'gray',
    'orange',
    'yellow',
    'tan',
    'silver',
    'green',
    'white',
    'pink',
    'cyan',
    'purple',
    'lime',
    'mauve',
    'ochre',
    'iceblue',
    'black',
    'yellow2',
    'yellow3',
    'green2',
    'green3',
    'cyan2',
    'cyan3',
    'blue2',
    'blue3',
    'violet',
    'violet2',
    'magenta',
    'magenta2',
    'red2',
    'red3',
    'orange2',
    'orange3',
]

d = {3: [0, 2, 3, 5, 6, 19, 21, 38, 45, 53, 77, 80, 82],
    0 : [1, 4, 7, 8, 9, 11, 12, 15, 29, 34, 35],
    2 : [10, 17, 20, 23, 25, 30, 70],
    1 : [13, 14, 16, 28, 40, 41, 65, 66], 7: [18, 42, 43, 44, 55],
    4 : [22, 26, 27, 71], 5: [24, 31, 32, 33, 47, 54], 6: [36, 37, 59],
    13: [39, 75, 78, 79], 11: [46, 51, 52, 57, 58, 61, 72, 76, 84],
    8 : [48, 49, 50, 56], 9: [60, 64], 10: [62, 63, 67],
    12: [68, 69, 73], 14: [74, 81, 83, 85, 88], 15: [86, 87, 89]}
for k in d:
    if k > 32:
        continue
    path_clusters = {}
    for i in d[k]:
        path_clusters[i] = colors[k]

    params = VisualizeSuboptimalPaths(
        suboptimal_paths=suboptimal_paths,
        radii=(0.3, 0.3),
        path_clusters=path_clusters,
        src_node_sphere={'color': 'blue', 'radius': 2.0},
        sink_node_sphere={'color': 'red', 'radius': 2.0},
    )

    tcl = draw_suboptimal_paths(
        parameters=params,
        tcl=tcl,
    )

    save_tcl(
        tcl=tcl,
        tcl_filename=f'clusters/cluster_{k}.tcl',
    )
