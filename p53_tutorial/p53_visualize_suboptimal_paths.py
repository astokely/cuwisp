import os

from cuwisp.paths import SuboptimalPaths
from cuwisp.visualize import draw_suboptimal_paths
from cuwisp.visualize import VisualizeSuboptimalPaths
from cuwisp.vmdtcl import save_tcl

# Calculation nam
calc = 'p53'

# Create a SuboptimalPaths class object.
suboptimal_paths = SuboptimalPaths()

'''
Calculation root directory (os.getcwd() returns the path to the 
current working directory).
'''
calc_dir = f'{os.getcwd()}/p53'

'''
Directory the spline files are in.
'''
splines_dir = f'{calc_dir}/p53Splines'

'''
Directory where the suboptimal paths visualization
tcl scripts are written to.
'''
graphics_dir = f'{calc_dir}/p53Graphics'

'''
Name of the xml file the calculation SuboptimalPaths
class object is serialized to.
'''
p53_suboptimal_paths_fname = f'{calc_dir}/p53SuboptimalPaths.xml'

'''
Deserialize the calculations SuboptimalPaths class object
'''
suboptimal_paths.deserialize(p53_suboptimal_paths_fname)

# Generate suboptimal paths tcl graphics scripts
for i in range(10):
    '''
    Trajectory frame index the suboptimal paths tcl 
    script is generated for.
    '''
    frame_index = i * 10

    '''
    Name of the tcl graphics script
    '''
    tcl_fname = f'{graphics_dir}/{calc}_frame{frame_index}.tcl'
    for path in suboptimal_paths.paths:
        '''
        Directory where the spline files for the current frame
        are located.
        '''
        spline_dir = f'{splines_dir}/frame{frame_index}'

        '''
        Name of the serialized spline object xml file.
        '''
        spline_fname = f'{spline_dir}/spline{path.index}.xml'
        path.serialized_splines = [
            spline_fname
        ]
    '''
    Visualization parameters:
    radii: Sets the path radii range, where the
        shortest paths will have the largest radii and
        vice verse. 
    color: Sets the path color gradient. The paths are colored
        based on length, where the shortest path will be the 
        left color and the longest path will be the right color. 
    src_node_sphere: If set, the src node will be represented by
        a sphere with the provided parameters.    
    sink_node_sphere: If set, the sink node will be represented by
        a sphere with the provided parameters.    
    path_indices: Indices of paths that visualization code are 
        generate for. 
    '''
    params = VisualizeSuboptimalPaths(
        suboptimal_paths=suboptimal_paths,
        radii=(0.1, 0.7),
        color=('red', 'green'),
        src_node_sphere={
            'color': 'blue',
            'radius': 2.0,
            'resolution': 250
        },
        sink_node_sphere={
            'color': 'red',
            'radius': 2.0,
            'resolution': 250
        },
        path_indices=list(range(len(suboptimal_paths.paths))),
    )

    '''
    parameters: Parameters uses for generating path visualization
        tcl.
    tcl: Empty string the tcl code is written to.
    '''
    tcl = ''
    tcl = draw_suboptimal_paths(
        parameters=params,
        tcl=tcl,
    )

    '''
    tcl: tcl string that the visualization code was written to.
    tcl_filename: Name of the file the tcl visualization code
        is written to.
    '''
    save_tcl(
        tcl=tcl,
        tcl_filename=tcl_fname
    )
