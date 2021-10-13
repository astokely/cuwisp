from cuwisp import merge_suboptimal_paths

calc_name='p53'
directory = calc_name

merge_suboptimal_paths(
    directory=directory,
    rounds=[_ for _ in range(5)],
    nodes_fname=f'{calc_name}/{calc_name}_nodes.xml',
    suboptimal_paths_fname=f'{calc_name}/'
                           f'{calc_name}_suboptimal_paths.xml'
)
