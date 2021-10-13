from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import os
import sys
from typing import (
    Optional,
    Tuple,
    List,
    Dict,
    Any,
)
from abserdes import Serializer as serializer

from .correlation_matrix import get_correlation_matrix
from .paths import (
    get_suboptimal_paths,
    SuboptimalPaths,
    Path,
    Edge,
    Nodes,
    built_in_rules,
    Rule,
)

from .cuwispio import (
    IO,
    naming_conventions,
)

def set_class_instance_attr(
        value: Any,
        default_value: Any,
) -> Any:
    if not value:
        return default_value
    return value

class CorrelationMatrixCalculation(serializer):

    def __init__(
            self,
            cuwisp_io: Optional[IO] = False,
            contact_map_distance_cutoff: Optional[float] = False,
            cuda_parameters: Optional[Tuple] = False,
            num_multiprocessing_processes: Optional[int] = 10,
            node_coordinate_frames: Optional[List[int]] = False,
    ) -> None:
        """
        @param cuwisp_io:
        @type: IO

        @param contact_map_distance_cutoff:
        @type contact_map_distance_cutoff: float

        @param cuda_parameters:
        @type cuda_parameters: tuple, optional

        @param num_multiprocessing_processes:
        @type num_multiprocessing_processes: int, optional

        @param node_coordinate_frames:
        @type node_coordinate_frames: list, optional

        @return:
        @rtype: None

        """
        if not cuwisp_io or not contact_map_distance_cutoff:
            return
        self.cuwisp_io = set_class_instance_attr(
            value=cuwisp_io,
            default_value=False,
        )
        self.contact_map_distance_cutoff = set_class_instance_attr(
            value=contact_map_distance_cutoff,
            default_value=False,
        )
        self.cuda_parameters = set_class_instance_attr(
            value=cuda_parameters,
            default_value=(256, 10, 256, 100,),
        )
        self.num_multiprocessing_processes = set_class_instance_attr(
            value=num_multiprocessing_processes,
            default_value=10,
        )
        self.node_coordinate_frames = set_class_instance_attr(
            value=node_coordinate_frames,
            default_value=False,
        )

    def calculate_correlation_matrix(
            self,
    ) -> None:
        """
        @return:
        @rtype: None

        """
        temp_file_directory = (
                os.path.dirname(
                    os.path.abspath(
                        sys.modules[
                            get_correlation_matrix.__module__].__file__
                    )
                ) + "/tmp"
        )
        (
            threads_per_block_com_calc,
            num_blocks_com_calc,
            threads_per_block_sum_coordinates_calc,
            num_blocks_sum_coordinates_calc
        ) = self.cuda_parameters

        get_correlation_matrix(
            cuwisp_io=self.cuwisp_io,
            contact_map_distance_limit=(
                self.contact_map_distance_cutoff
            ),
            temp_file_directory=temp_file_directory,
            threads_per_block_com_calc=threads_per_block_com_calc,
            num_blocks_com_calc=num_blocks_com_calc,
            threads_per_block_sum_coordinates_calc=(
                threads_per_block_sum_coordinates_calc
            ),
            num_blocks_sum_coordinates_calc=(
                num_blocks_sum_coordinates_calc
            ),
            num_multiprocessing_processes=(
                self.num_multiprocessing_processes
            ),
            node_coordinate_frames=self.node_coordinate_frames,
        )

def get_path_finding_rules(
        path_finding_rules: Dict,
) -> Dict:
    if not path_finding_rules:
        return built_in_rules()
    if path_finding_rules:
        rules = built_in_rules()
        for index, rule in path_finding_rules.items():
            rules[index] = Rule(*rule)
    return path_finding_rules

def get_output_suffix(
        simulation_rounds: List,
) -> str:
    suffix = f''
    for simulation_round in simulation_rounds:
        suffix += f'{simulation_round}_'
    return suffix[:-1]

def initialize_cuwispio_list_attrs(
        cuwisp_io: IO,
) -> IO:
    if not cuwisp_io.suboptimal_paths_fnames:
        cuwisp_io.suboptimal_paths_fnames = []
    if not cuwisp_io.serialized_correlation_matrices_directories:
        cuwisp_io.serialized_correlation_matrices_directories = []
    if not cuwisp_io.serialized_suboptimal_paths_directories:
        cuwisp_io.serialized_suboptimal_paths_directories = []
    return cuwisp_io

def create_backup_output_sp_serialization_directories(
        directory: str,
) -> None:
    if not os.path.exists(path=f'{directory}/backup'):
        os.makedirs(name=f'{directory}/backup')
    if not os.path.exists(path=f'{directory}/output'):
        os.makedirs(name=f'{directory}/output')

def _create_sp_calc_serialization_directories(
        cuwisp_io: IO,
        directory: str,
        existing_directories: List,
        directory_type: str,
) -> int:
    if directory_type == 'correlation_matrices':
        if not os.path.exists(path=directory):
            os.makedirs(name=directory)
        if not directory in existing_directories:
            cuwisp_io.serialized_correlation_matrices_directories. \
                append(directory)
        return cuwisp_io.serialized_correlation_matrices_directories. \
            index(directory)
    elif directory_type == 'suboptimal_paths':
        if not os.path.exists(path=directory):
            os.makedirs(name=directory)
        create_backup_output_sp_serialization_directories(
            directory=directory,
        )
        if not directory in existing_directories:
            cuwisp_io.serialized_suboptimal_paths_directories. \
                append(directory)
        return cuwisp_io.serialized_suboptimal_paths_directories. \
            index(directory)

def create_sp_calc_correlation_matrices_serialization_directories(
        cuwisp_io: IO,
        suffix: str,
) -> int:
    existing_cmsds = (
        cuwisp_io.serialized_correlation_matrices_directories
    )
    cmsd_basename = (
        naming_conventions()[
            cuwisp_io.naming_convention
        ](
            'serialized',
            'correlation',
            'matrices',
        )
    )
    cmsd = (
            f'{cuwisp_io.suboptimal_paths_serialization_directory}/'
            + cmsd_basename
            + f'_{suffix}'
    )
    directory_index = _create_sp_calc_serialization_directories(
        cuwisp_io=cuwisp_io,
        directory=cmsd,
        existing_directories=existing_cmsds,
        directory_type='correlation_matrices'
    )
    return directory_index

def create_sp_calc_suboptimal_paths_serialization_directories(
        cuwisp_io: IO,
        suffix: str,
) -> int:
    existing_spsds = (
        cuwisp_io.serialized_suboptimal_paths_directories
    )
    spsd_basename = (
        naming_conventions()[
            cuwisp_io.naming_convention
        ](
            'serialized',
            'suboptimal',
            'paths',
        )
    )
    spsd = (
            f'{cuwisp_io.suboptimal_paths_serialization_directory}/'
            + spsd_basename
            + f'_{suffix}'
    )
    directory_index = _create_sp_calc_serialization_directories(
        cuwisp_io=cuwisp_io,
        directory=spsd,
        existing_directories=existing_spsds,
        directory_type='suboptimal_paths'
    )
    return directory_index

def create_sp_calc_serialization_directories(
        cuwisp_io: IO,
        suffix: str,
) -> Tuple:
    i = create_sp_calc_correlation_matrices_serialization_directories(
        cuwisp_io=cuwisp_io,
        suffix=suffix,
    )
    j = create_sp_calc_suboptimal_paths_serialization_directories(
        cuwisp_io=cuwisp_io,
        suffix=suffix,
    )
    return i, j

def get_restart_correlation_matrix_directory(
        cuwisp_io: IO,
        suffix: str,
) -> str:
    spsd = cuwisp_io.suboptimal_paths_serialization_directory
    for d in os.listdir(spsd):
        is_restart_directory = d.split(suffix) == [
            'serialized_correlation_matrices_', ''
        ]
        if is_restart_directory:
            return f'{spsd}/{d}'

def _get_restart_correlation_matrix_fname(
        restart_correlation_matrix_directory: str,
        start_round: int,
) -> str:
    for f in os.listdir(restart_correlation_matrix_directory):
        if f.split('_').pop()[:-4] == f'{start_round}':
            return (
                f'{restart_correlation_matrix_directory}/{f}'
            )

def get_restart_correlation_matrix_fname(
        cuwisp_io: IO,
        start_round: int,
        suffix: str,
) -> List:
    restart_correlation_matrix_directory = (
        get_restart_correlation_matrix_directory(
            cuwisp_io=cuwisp_io,
            suffix=suffix,
        )
    )
    restart_correlation_matrix_fname = (
        _get_restart_correlation_matrix_fname(
            restart_correlation_matrix_directory=(
                restart_correlation_matrix_directory
            ),
            start_round=start_round
        )
    )
    return restart_correlation_matrix_fname

def get_num_existing_sp_calc_output_files(
        sposd: str,
) -> int:
    num_existing_sp_output_files = 0
    for _ in os.listdir(sposd):
        num_existing_sp_output_files += 1
    return num_existing_sp_output_files

def get_calc_sp_output_serialization_directory(
        cuwisp_io: IO,
        suffix: str,
) -> str:
    spsd = cuwisp_io.suboptimal_paths_serialization_directory
    for d in os.listdir(spsd):
        is_calc_sp_serialization_directory = d.split(suffix) == [
            'serialized_suboptimal_paths_', ''
        ]
        if is_calc_sp_serialization_directory:
            return f'{spsd}/{d}/output'

def set_suboptimal_paths_calc_fname(
        cuwisp_io: IO,
        suffix: str,
) -> IO:
    calc_sposd = (
        get_calc_sp_output_serialization_directory(
            cuwisp_io=cuwisp_io,
            suffix=suffix,
        )
    )
    num_existing_sp_output_files = (
        get_num_existing_sp_calc_output_files(
            sposd=calc_sposd,
        )
    )
    sp_basename_no_ext = os.path.basename(
        cuwisp_io.suboptimal_paths_fname
    )[:-4]
    sp_calc_fname = (
            f'{calc_sposd}/{sp_basename_no_ext}'
            + f'_{suffix}_{num_existing_sp_output_files}.xml'
    )
    cuwisp_io.suboptimal_paths_fnames.append(sp_calc_fname)
    return cuwisp_io

class SuboptimalPathsCalculation(serializer):

    def __init__(
            self,
            cuwisp_io: IO,
            src: int,
            sink: int,
            cutoff: Optional[float] = False,
            threads_per_block: Optional[int] = 256,
            use_contact_map_correlation_matrix: Optional[bool] = True,
            serialization_frequency: Optional[float] = False,
            simulation_rounds: Optional[List[int]] = False,
            gpu: Optional[int] = 0,
            max_num_paths: Optional[int] = 25,
            path_finding_rules: Optional[Dict] = False,
            restart: Optional[int] = False,
    ) -> None:
        """
        @param cuwisp_io:
        @type: IO

        @param src:
        @type src: int

        @param sink:
        @type sink: int

        @param cutoff:
        @type cutoff: float, optional

        @param threads_per_block:
        @type threads_per_block: int, optional

        @param use_contact_map_correlation_matrix:
        @type use_contact_map_correlation_matrix: bool, optional

        @param serialization_frequency:
        @type serialization_frequency: float, optional

        @param simulation_rounds:
        @type simulation_rounds: list, optional

        @param gpu:
        @type gpu: int, optional

        @param max_num_paths:
        @type max_num_paths: int, optional

        @param path_finding_rules:
        @type path_finding_rules: dict, optional

        @param restart
        @type: int, optional

        @return:
        @rtype: None

        """
        if not cuwisp_io or not src or not sink:
            return
        self.cuwisp_io = set_class_instance_attr(
            value=cuwisp_io,
            default_value=False,
        )
        self.src = set_class_instance_attr(
            value=src,
            default_value=False,
        )
        self.sink = set_class_instance_attr(
            value=sink,
            default_value=False,
        )
        self.cutoff = set_class_instance_attr(
            value=cutoff,
            default_value=False,
        )
        self.threads_per_block = set_class_instance_attr(
            value=threads_per_block,
            default_value=256,
        )
        self.use_contact_map_correlation_matrix = (
            set_class_instance_attr(
                value=use_contact_map_correlation_matrix,
                default_value=True
            )
        )
        self.serialization_frequency = set_class_instance_attr(
            value=serialization_frequency,
            default_value=False,
        )
        self.simulation_rounds = set_class_instance_attr(
            value=simulation_rounds,
            default_value=False,
        )
        self.gpu = set_class_instance_attr(
            value=gpu,
            default_value=0,
        )

        self.max_num_paths = set_class_instance_attr(
            value=max_num_paths,
            default_value=25,
        )
        self.path_finding_rules = set_class_instance_attr(
            value=path_finding_rules,
            default_value=False,
        )
        self.restart = set_class_instance_attr(
            value=restart,
            default_value=False,
        )

    def calculate_suboptimal_paths(
            self,
    ) -> None:
        """"
        @return:
        @rtype: None

        """
        if not self.simulation_rounds:
            self.simulation_rounds = [0, 1, 2, 3, 4]
        rules = get_path_finding_rules(
            path_finding_rules=self.path_finding_rules
        )
        suffix = get_output_suffix(
            simulation_rounds=self.simulation_rounds,
        )
        initialize_cuwispio_list_attrs(
            cuwisp_io=self.cuwisp_io
        )
        cmsd_index, spsd_index = (
            create_sp_calc_serialization_directories(
                cuwisp_io=self.cuwisp_io,
                suffix=suffix,
            )
        )
        set_suboptimal_paths_calc_fname(
            cuwisp_io=self.cuwisp_io,
            suffix=suffix
        )
        simulation_rounds_ = self.simulation_rounds
        correlation_matrix_fname = False
        if not isinstance(self.restart, bool):
            if isinstance(self.restart, list):
                simulation_rounds_ = self.restart
            else:
                simulation_rounds_ = self.simulation_rounds[
                self.simulation_rounds.index(self.restart):
                ]
            correlation_matrix_fname = (
                get_restart_correlation_matrix_fname(
                    cuwisp_io=self.cuwisp_io,
                    start_round=simulation_rounds_[0],
                    suffix=suffix,
                )
            )
        if not correlation_matrix_fname:
            if self.use_contact_map_correlation_matrix:
                correlation_matrix_fname = (
                    self.cuwisp_io.contact_map_correlation_matrix_fname
                )
            else:
                correlation_matrix_fname = (
                    self.cuwisp_io.contact_map_correlation_matrix_fname
                )
        self.cuwisp_io.serialize(
            xml_filename=self.cuwisp_io.io_fname
        )
        get_suboptimal_paths(
            cuwisp_io=self.cuwisp_io,
            src=self.src,
            sink=self.sink,
            cutoff=self.cutoff,
            threads_per_block=self.threads_per_block,
            serialization_frequency=self.serialization_frequency,
            correlation_matrix_serialization_directory=(
                self.cuwisp_io.
                    serialized_correlation_matrices_directories[
                     cmsd_index
                    ]
            ),
            suboptimal_paths_serialization_directory=(
                    self.cuwisp_io.
                        serialized_suboptimal_paths_directories[
                            spsd_index
                        ] + f'/backup'
            ),
            simulation_rounds=simulation_rounds_,
            gpu_index=self.gpu,
            max_num_paths=self.max_num_paths,
            rules=rules,
            correlation_matrix_fname=correlation_matrix_fname,
        )

def get_suboptimal_path_calcs_xmls(
        xmls: List[str],
        path_finding_rounds: List[int],
) -> List[str]:
    suboptimal_path_calcs_xmls = []
    for xml in xmls:
        for path_finding_round in path_finding_rounds:
            if path_finding_round in xml and 'nodes' not in xml:
                suboptimal_path_calcs_xmls.append(xml)
    return list(set(suboptimal_path_calcs_xmls))

def get_serialized_suboptimal_paths(
        directory: str,
        path_finding_rounds: List[int],
) -> List[str]:
    """
    @param directory:
    @type directory: str

    @param path_finding_rounds:
    @type path_finding_rounds: list

    @return:
    @rtype: list

    """
    xmls = [
        os.path.abspath(os.path.join(directory, f)) for f
        in os.listdir(directory) if 'xml' in f
    ]
    path_finding_rounds = [
        str(path_finding_round)
        for path_finding_round in path_finding_rounds

    ]
    return get_suboptimal_path_calcs_xmls(
        xmls=xmls,
        path_finding_rounds=path_finding_rounds
    )

def deserialize_suboptimal_paths(
        suboptimal_path_xmls: List[str],
) -> List[SuboptimalPaths]:
    """
    @param suboptimal_path_xmls:
    @type suboptimal_path_xmls: list

    @return:
    @rtype: list

    """
    suboptimal_paths = []
    suboptimal_path_xmls = set(suboptimal_path_xmls)
    for xml in suboptimal_path_xmls:
        suboptimal_path = SuboptimalPaths()
        suboptimal_path.deserialize(xml)
        suboptimal_paths.append(suboptimal_path)
    return suboptimal_paths

def get_sorted_suboptimal_paths_dict(
        suboptimal_paths: SuboptimalPaths,
) -> Dict[float, List[Edge]]:
    """
    @param suboptimal_paths:
    @type suboptimal_paths: SuboptimalPaths

    @return:
    @rtype: dict

    """
    suboptimal_paths_dict = {
        path.length: path.edges
        for suboptimal_paths_obj
        in suboptimal_paths
        for path in suboptimal_paths_obj.paths
    }
    return dict(
        sorted(
            suboptimal_paths_dict.items()
        )
    )

def get_src_sink(
        suboptimal_paths: SuboptimalPaths,
) -> Tuple[int]:
    """
    @param suboptimal_paths:
    @type suboptimal_paths: SuboptimalPaths

    @return:
    @rtype: tuple

    """
    return (
        suboptimal_paths.src,
        suboptimal_paths.sink,
    )

def merge_suboptimal_paths(
        directory: str,
        rounds: List[int],
        nodes_fname: str,
        suboptimal_paths_fname: str,
) -> None:
    """
    @param directory:
    @type directory: str

    @param rounds:
    @type rounds: list

    @param nodes_fname:
    @type nodes_fname: str

    @param suboptimal_paths_fname:
    @type suboptimal_paths_fname: str

    @return:
    @rtype: None

    """
    suboptimal_paths = (
        get_serialized_suboptimal_paths(
            directory,
            rounds,
        )
    )
    suboptimal_paths_objs = (
        deserialize_suboptimal_paths(
            suboptimal_paths
        )
    )
    src, sink = get_src_sink(
        suboptimal_paths_objs[0]
    )
    suboptimal_paths_dict = (
        get_sorted_suboptimal_paths_dict(
            suboptimal_paths_objs
        )
    )
    suboptimal_paths = SuboptimalPaths()
    nodes_obj = Nodes()
    nodes_obj.deserialize(nodes_fname)
    path_index = 0
    for path_length in suboptimal_paths_dict:
        path = Path()
        path.length = path_length
        path.edges = []
        path_nodes = set([])
        for path_edge in suboptimal_paths_dict[path_length]:
            node1_index = path_edge.node1.index
            node2_index = path_edge.node2.index
            path_nodes.add(node1_index)
            path_nodes.add(node2_index)
            edge = Edge()
            edge.node1 = nodes_obj[node1_index]
            edge.node2 = nodes_obj[node2_index]
            path.edges.append(edge)
        path.src = src
        path.sink = sink
        path.index = path_index
        path.num_nodes = len(path_nodes)
        path.num_edges = len(path.edges)
        suboptimal_paths.paths.append(path)
        path_index += 1
    suboptimal_paths.src = src
    suboptimal_paths.sink = sink
    suboptimal_paths.num_paths = len(suboptimal_paths.paths)
    suboptimal_paths.serialize(suboptimal_paths_fname)
