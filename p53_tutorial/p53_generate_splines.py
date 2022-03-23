import os

import numpy as np

from cuwisp import SuboptimalPaths
from cuwisp.splines import generate_splines


# Calculation root directory
p53_suboptimal_paths_dir = f'{os.getcwd()}/p53'

# Suboptimal paths xml filename
p53_suboptimal_paths_fname = (
    f'{p53_suboptimal_paths_dir}/p53SuboptimalPaths.xml'
)

# Directory where spline numpy binary files are saved.
p53_splines_dir = f'{p53_suboptimal_paths_dir}/p53Splines'

'''
Number of frames the node coordinates are saved for.
Since this trajectory is 100 frames, the node coordinates
are saved for every frame.
'''
n_frames = 100

'''
Frame indices splines are generated for. The below code is 
equivalent to spline_frames = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
so a spline is generated for every ten frames.
'''
spline_frames = list(
    np.arange(
        0, n_frames, 10
    )
)

# Create a SuboptimalPaths class object.
p53_suboptimal_paths = SuboptimalPaths()

# Deserialize the calculation's SuboptimalPaths object.
p53_suboptimal_paths.deserialize(
    p53_suboptimal_paths_fname
)

# Generate splines using the parameters set above.
generate_splines(
    suboptimal_paths=p53_suboptimal_paths,
    suboptimal_paths_fname=p53_suboptimal_paths_fname,
    frames=spline_frames,
    output_directory=p53_splines_dir
)
