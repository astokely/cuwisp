from cuwisp.paths import SuboptimalPaths
import cuwisp.cluster as analysis
from cuwisp.splines import Spline
from cuwisp.splines import generate_splines

sp = SuboptimalPaths()
sp.deserialize('p53/p53_suboptimal_paths.xml')


path_splines = [
    Spline().deserialize(path.serialized_splines[0])
    for path in sp.paths
]
i = 0.8
while i <=1.0:
    area_matrix = analysis.get_area_between_paths_matrix(
        path_splines, degree=3, upper_bound=i, lower_bound=0.5
    )
    path_clusters = analysis.cluster_paths(area_matrix)
    print(path_clusters)
    print('')
    i += 0.1

