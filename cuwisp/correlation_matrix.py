from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import os
import shutil
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing import sharedctypes
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import mdtraj as md
import numpy as np

# noinspection PyUnresolvedReferences
from .calccom import calc_com as calc_com
from .cparse.cparse import parsePdb as parse
from .cuwispio import IO
from .nodes import Node
from .nodes import Nodes
from .numba_cuda.cuda_contact_map import cuda_contact_map
from .numba_cuda.cuda_correlation_matrix import cuda_correlation_matrix
from .numba_cuda.hollow_matrix import hollowMatrix
from .numba_cuda.sum_coordinates import sumCoords

NODES = defaultdict(list)

def ctypes_matrix(
        n: int,
        m: int,
) -> Any:
    """
    @param n:
    @type n: int

    @param m:
    @type m: int

    @return:
    @rtype: object

    """
    return np.ctypeslib.as_ctypes(
        np.array(
            [
                np.ctypeslib.as_ctypes(np.zeros(n, dtype=np.float64))
                for _ in range(m)
            ]
        )
    )


# noinspection PyProtectedMember
def shared_ctypes_multiprocessing_array(
        ctypes_array: Any,
) -> Any:
    """
    @param ctypes_array:
    @type ctypes_array: object

    @return:
    @rtype: object

    """
    return sharedctypes.RawArray(
        ctypes_array._type_,
        ctypes_array
    )


class Molecule:

    def __init__(
            self,
            trajectory: Optional[md.Trajectory] = False,
            atomnames: Optional[np.ndarray] = False,
            chains: Optional[np.ndarray] = False,
            masses: Optional[np.ndarray] = False,
            resids: Optional[np.ndarray] = False,
            resnames: Optional[np.ndarray] = False,
            segment_ids: Optional[np.ndarray] = False,
            elements: Optional[np.ndarray] = False,
            coordinates: Optional[np.ndarray] = False,
            node_tag_to_atom_indices: Optional[defaultdict] = False,
            node_tags_in_order: Optional[np.ndarray] = False,
            nodes_array: Optional[np.ndarray] = False,
    ) -> None:
        """
        @param trajectory:
        @type trajectory: mdtraj.Trajectory, optional

        @param atomnames:
        @type atomnames: numpy.ndarray, optional

        @param chains:
        @type chains: numpy.ndarray, optional

        @param masses:
        @type masses: numpy.ndarray, optional

        @param resids:
        @type resids: numpy.ndarray, optional

        @param resnames:
        @type resnames: numpy.ndarray, optional

        @param segment_ids:
        @type segment_ids: numpy.ndarray, optional

        @param elements:
        @type elements:

        @param coordinates:
        @type coordinates: numpy.ndarray, optional

        @param node_tag_to_atom_indices:
        @type node_tag_to_atom_indices: defaultdict

        @param node_tags_in_order:
        @type node_tags_in_order: numpy.ndarray, optional

        @param nodes_array:
        @type nodes_array: numpy.ndarray, optional

        @return
        @rtype: None

        """
        self.trajectory = trajectory
        self.atomnames = atomnames
        self.chains = chains
        self.masses = masses
        self.resids = resids
        self.resnames = resnames
        self.segment_ids = segment_ids
        self.elements = elements
        self.coordinates = coordinates
        self.node_tag_to_atom_indices = node_tag_to_atom_indices
        self.node_tags_in_order = node_tags_in_order
        self.nodes_array = nodes_array
        return

    def parse(
            self,
            traj: md.Trajectory,
            num_frames: Optional[bool] = False,
            frame: Optional[bool] = False,
    ) -> None:
        """
        @param traj:
        @type traj: mdtraj.Trajectory

        @param num_frames:
        @type num_frames: bool, optional

        @param frame:
        @type frame: bool, optional

        @return:
        @rtype: None

        """
        self.trajectory = traj
        top = self.trajectory.topology
        atoms = [atom for atom in top.atoms]
        coords = [10 * xyz for xyz in self.trajectory[0].xyz][0]
        if num_frames:
            for i in range(len(coords)):
                scmat[i][frame] = coords[i][0]
                scmat[i][num_frames + frame] = coords[i][1]
                scmat[i][2 * num_frames + frame] = coords[i][2]

        self.atomnames = np.array(
            [
                atom.name for atom in atoms
            ]
        )
        self.chains = np.array(
            [
                atom.residue.chain.index for atom in atoms
            ]
        )
        self.masses = np.array(
            [
                atom.element.mass for atom in atoms
            ]
        )
        self.resids = np.array(
            [
                atom.residue.index for atom in atoms
            ]
        )
        self.resnames = np.array(
            [
                atom.residue.name for atom in atoms
            ]
        )
        self.segment_ids = np.array(
            [
                atom.residue.segment_id for atom in atoms
            ]
        )
        self.elements = np.array(
            [
                atom.element.symbol for atom in atoms
            ]
        )
        self.coordinates = np.array(coords, np.float64)

    def map_atoms_to_node_tags(
            self
    ) -> None:
        """
        @return
        @rtype: None

        """
        indices_tags_dict = {
            index: (
                    f'{self.chains[index]}_'
                    + f'{self.resnames[index]}_'
                    + f'{self.resids[index]}_'
                    + f'{self.segment_ids[index]}'
            )
            if self.segment_ids[index] != '' else (
                    f'{self.chains[index]}_'
                    + f'{self.resnames[index]}_'
                    + f'{self.resids[index]}'
            )
            for index in range(len(self.coordinates))
        }
        self.node_tags_in_order = np.array(
            list(
                {
                    tag: None
                    for tag in
                    indices_tags_dict.values()
                }
            )
        )
        self.node_tag_to_atom_indices = defaultdict(list)
        for atom_index, tag in indices_tags_dict.items():
            self.node_tag_to_atom_indices[tag].append(
                atom_index
            )

    def map_nodes_to_node_tags(
            self,
            coms: np.ndarray,
    ) -> None:
        """
        @param coms:
        @type coms: numpy.ndarray

        @return:
        @rtype: None

        """
        self.nodes_array = np.empty(
            (
                len(self.node_tags_in_order),
                3
            )
        )
        for index in range(len(coms)):
            self.nodes_array[index][0] = coms[index][0]
            self.nodes_array[index][1] = coms[index][1]
            self.nodes_array[index][2] = coms[index][2]


def create_symbolic_link(
        src_file_path: str,
        symbolic_link_path: str,
) -> None:
    process = subprocess.Popen(
        [
            f'ln', f'-s',
            f'{src_file_path}',
            f'{symbolic_link_path}',
        ],
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    out, err = process.communicate()
    errcode = process.returncode
    process.kill()
    process.terminate()


def remove_symbolic_link(
        src_file_path: str,
        symbolic_link_path: str,
) -> None:
    process = subprocess.Popen(
        [
            f'unlink',
            f'{symbolic_link_path}',
        ],
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    out, err = process.communicate()
    errcode = process.returncode
    process.kill()
    process.terminate()


def catdcd(
        catdcd_exe_dir: str,
        input_dcd_filename: str,
        topology_filename: str,
        output_pdb_filename: str,
) -> None:
    create_symbolic_link(
        f'{catdcd_exe_dir}/libexpat.so.0',
        f'libexpat.so.0',
    )
    process = subprocess.Popen(
        [
            f'{catdcd_exe_dir}/catdcd',
            '-o', f'{output_pdb_filename}',
            '-otype', 'pdb',
            '-s', f'{topology_filename}',
            f'{input_dcd_filename}'
        ],
        shell=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    out, err = process.communicate()
    errcode = process.returncode
    process.kill()
    process.terminate()
    remove_symbolic_link(
        f'{catdcd_exe_dir}/libexpat.so.0',
        f'libexpat.so.0',
    )


def parse_pdb(
        args: Tuple[Union[int, int, str]],
) -> Tuple[Union[int, Molecule]]:
    """
    @param args:
    @type args: tuple

    @return:
    @rtype: tuple

    """
    path, pdb_file, num_frames = args
    frame = int(pdb_file[:-4])
    traj = md.load(path + pdb_file)
    pdb = Molecule()
    pdb.parse(traj, num_frames, frame)
    return (
        frame,
        pdb
    )


def parse_dcd(
        input_dcd_filename: str,
        topology_filename: str,
        output_pdb_filename: str,
        output_tmp_pdbs_directory: str,
) -> None:
    """
    @param input_dcd_filename:
    @type input_dcd_filename: str

    @param topology_filename:
    @type topology_filename: str

    @param output_pdb_filename:
    @type output_pdb_filename: str

    @param output_tmp_pdbs_directory:
    @type output_tmp_pdbs_directory: str

    @return:
    @rtype: None

    """
    catdcd_exe_dir = (
            os.path.dirname(
                os.path.abspath(
                    f'{sys.modules[Nodes.__module__].__file__}'
                )
            ) + f'/bin'
    )
    if os.path.exists(output_pdb_filename):
        os.remove(output_pdb_filename)
    if os.path.exists(output_tmp_pdbs_directory):
        shutil.rmtree(output_tmp_pdbs_directory)
    os.makedirs(output_tmp_pdbs_directory)
    catdcd(
        catdcd_exe_dir,
        input_dcd_filename,
        topology_filename,
        output_pdb_filename,
    )
    parse(
        output_pdb_filename,
        f'{output_tmp_pdbs_directory}/',
    )
    os.remove(output_pdb_filename)


def prepare_trajectory_for_analysis(
        temp_file_directory: str,
        cuwisp_io: IO,
) -> List[str]:
    """
    @param temp_file_directory:
    @type temp_file_directory: str

    @param cuwisp_io:
    @type: IO

    @return:
    @rtype: list

    """
    if os.path.exists(temp_file_directory):
        shutil.rmtree(temp_file_directory)
    os.makedirs(temp_file_directory)
    if not cuwisp_io.topology_fname:
        parse(
            cuwisp_io.trajectory_fname,
            temp_file_directory + "/"
        )
    else:
        parse_dcd(
            cuwisp_io.trajectory_fname,
            cuwisp_io.topology_fname,
            f'{temp_file_directory}/tmp.pdb',
            temp_file_directory,
        )
    pdb_single_frame_files = [
        pdb_file for pdb_file
        in os.listdir(temp_file_directory + "/")
    ]
    return pdb_single_frame_files


def get_parameters_for_multiprocessing_pdb_parser(
        temp_file_directory: str,
        pdb_from_trajectory: Molecule,
) -> Tuple[Union[int, int, List[str], int]]:
    """
    @param temp_file_directory:
    @type temp_file_directory: str

    @param pdb_from_trajectory:
    @type pdb_from_trajectory: Molecule

    @return:
    @rtype: tuple

    """
    num_traj_frames = len(
        os.listdir(temp_file_directory + "/")
    )
    num_atoms = len(
        pdb_from_trajectory.coordinates
    )
    paths = [
        temp_file_directory + "/"
        for _ in range(num_traj_frames)
    ]
    num_frames = [
        num_traj_frames for _
        in range(num_traj_frames)
    ]
    return (
        num_traj_frames,
        num_atoms,
        paths,
        num_frames,
    )


def multiprocessing_pdb_parser(
        num_traj_frames: int,
        num_atoms: int,
        num_multiprocessing_processes: int,
        paths: List[str],
        pdb_single_frame_files: List[str],
        num_frames: int,
        average_pdb: Molecule,
        num_blocks_sum_coordinates_calc: int,
        threads_per_block_sum_coordinates_calc: int,
) -> List[Molecule]:
    """
    @param num_traj_frames:
    @type num_traj_frames: int

    @param num_atoms:
    @type num_atoms: int

    @param num_multiprocessing_processes:
    @type num_multiprocessing_processes: int

    @param paths:
    @type paths: list

    @param pdb_single_frame_files:
    @type pdb_single_frame_files: list

    @param num_frames:
    @type num_frames: int

    @param average_pdb:
    @type average_pdb: Molecule

    @param num_blocks_sum_coordinates_calc:
    @type num_blocks_sum_coordinates_calc: int

    @param threads_per_block_sum_coordinates_calc:
    @type threads_per_block_sum_coordinates_calc: int

    @return:
    @rtype: list

    """
    # noinspection PyGlobalUndefined
    global scmat
    cmat = ctypes_matrix(
        3 * num_traj_frames,
        num_atoms
    )
    scmat = shared_ctypes_multiprocessing_array(cmat)
    with Pool(num_multiprocessing_processes) as pool:
        pdbs = list(
            pool.map(
                parse_pdb,
                zip(
                    paths,
                    pdb_single_frame_files,
                    num_frames
                )
            )
        )
    pdbs = list(
        dict(
            sorted(
                {
                    pdb[0]: pdb[1] for pdb in pdbs
                }.items()
            )
        ).values()
    )
    coordinates = np.ctypeslib.as_array(scmat)
    del scmat
    average_pdb.coordinates = sumCoords(
        coordinates,
        num_traj_frames,
        num_atoms,
        num_blocks_sum_coordinates_calc,
        threads_per_block_sum_coordinates_calc
    )
    for pdb in pdbs:
        pdb.map_atoms_to_node_tags()
    average_pdb.map_atoms_to_node_tags()
    pdbs.append(average_pdb)
    return pdbs


def serialize_nodes(
        average_pdb: Molecule,
        coms: np.ndarray,
        cuwisp_io: IO,
        node_coordinate_frames: List[int],
) -> None:
    """
    @param average_pdb:
    @type average_pdb: Molecule

    @param coms:
    @type coms: numpy.ndarray

    @param cuwisp_io:
    @type: IO

    @param node_coordinate_frames:
    @type node_coordinate_frames: list

    @return:
    @rtype: None

    """
    nodes = Nodes()
    node_index = 0
    atom_indices_list = []
    for node_tag, atom_indices in \
            average_pdb.node_tag_to_atom_indices.items():
        node = Node()
        node.index = node_index
        node.atom_indices = atom_indices
        node.tag = node_tag
        node.segment_id = (
            f'{average_pdb.segment_ids[atom_indices[0]]}'
        )
        node.resname = (
            f'{average_pdb.resnames[atom_indices[0]]}'
        )
        node.chain_index = (
            average_pdb.chains[atom_indices[0]]
        )
        node.resid = (
            average_pdb.resids[atom_indices[0]]
        )
        nodes[node_index] = node
        node_index += 1
        node.coordinates_directory = (
            cuwisp_io.node_coordinates_directory
        )
        atom_indices_list.append(atom_indices)
        node.coordinate_frames = node_coordinate_frames
    nodes.num_nodes = node_index + 1
    for frame in node_coordinate_frames:
        np.save(
            (
                    f'{cuwisp_io.node_coordinates_directory}/'
                    + f'{frame}.npy'
            )
            , coms[frame]
        )
    nodes.serialize(cuwisp_io.nodes_fname)
    return atom_indices_list


def calculate_center_of_masses(
        pdbs: List[Molecule],
        threads_per_block_com_calc: int,
        num_blocks_com_calc: int,
) -> np.ndarray:
    """
    @param pdbs:
    @type pdbs: list

    @param threads_per_block_com_calc:
    @type threads_per_block_com_calc: int

    @param num_blocks_com_calc:
    @type num_blocks_com_calc: int

    @return:
    @rtype: numpy.ndarray

    """
    all_indices = []
    for pdb in pdbs:
        all_indices.append(
            [
                pdb.node_tag_to_atom_indices[node_tag]
                for node_tag
                in pdb.node_tags_in_order
            ]
        )

    all_coords = [
        pdb.coordinates for pdb in pdbs
    ]
    all_masses = [
        pdb.masses for pdb in pdbs
    ]
    all_coms = calc_com(
        all_indices,
        all_coords,
        all_masses,
        threads_per_block_com_calc,
        num_blocks_com_calc
    )
    return all_coms


def save_matrix(
        fname: str,
        a: np.ndarray,
) -> None:
    """
    @param fname:
    @type fname: str

    @param a:
    @type a: numpy.ndarray

    @return:
    @rtype: None

    """
    np.save(
        fname,
        a,
    )


def get_node_com_coordinates_array(
        num_nodes: int,
        num_pdbs: int,
        pdbs: List[Molecule],
) -> np.ndarray:
    """
    @param num_nodes:
    @type num_nodes: int

    @param num_pdbs:
    @type num_pdbs: int

    @param pdbs:
    @type pdbs: list

    @return:
    @rtype: numpy.ndarray

    """
    nodes = np.array(
        [
            np.zeros((num_pdbs, 3), dtype=np.float64)
            for _ in range(num_nodes)
        ]
        , dtype=object
    )
    for pdb_index in range(num_pdbs):
        for node_index in range(num_nodes):
            nodes[node_index][pdb_index][0] = (
                pdbs[pdb_index].nodes_array[node_index][0]
            )
            nodes[node_index][pdb_index][1] = (
                pdbs[pdb_index].nodes_array[node_index][1]
            )
            nodes[node_index][pdb_index][2] = (
                pdbs[pdb_index].nodes_array[node_index][2]
            )
    return nodes


def get_contact_map(
        correlation_matrix: np.ndarray,
        average_pdb: Molecule,
        contact_map_distance_cutoff: float,
        node_atom_indices: List,
) -> np.ndarray:
    """
    @param correlation_matrix:
    @type correlation_matrix: numpy.ndarray

    @param average_pdb:
    @type average_pdb: Molecule

    @param contact_map_distance_cutoff:
    @type contact_map_distance_cutoff: float

    @param node_atom_indices:
    @type node_atom_indices: list

    @return:
    @rtype: numpy.ndarray

    """
    contact_map = np.ones(correlation_matrix.shape)
    avg_coords = average_pdb.coordinates
    correlation_matrix_after_contact_map = np.zeros(
        correlation_matrix.shape,
        dtype=np.float64
    )
    if contact_map_distance_cutoff != np.inf:
        cuda_contact_map(
            correlation_matrix,
            correlation_matrix_after_contact_map,
            contact_map,
            contact_map_distance_cutoff,
            avg_coords,
            node_atom_indices
        )
        return contact_map, correlation_matrix_after_contact_map
    return contact_map, correlation_matrix


def _get_correlation_matrix(
        num_nodes: int,
        nodes: np.ndarray,
        average_pdb: Molecule,
) -> np.ndarray:
    """
    @param num_nodes:
    @type num_nodes: int

    @param nodes:
    @type nodes: numpy.ndarray

    @param average_pdb:
    @type average_pdb: Molecule

    @return:
    @rtype: numpy.ndarray

    """
    correlation_matrix = np.zeros(
        (
            num_nodes,
            num_nodes
        ), dtype=np.float64
    )
    h_correlation_matrix = np.zeros(
        (
            num_nodes,
            num_nodes
        ), dtype=np.float64
    )
    cuda_correlation_matrix(
        nodes,
        average_pdb.nodes_array,
        correlation_matrix
    )
    hollowMatrix(
        correlation_matrix,
        h_correlation_matrix
    )
    return h_correlation_matrix


def get_correlation_matrix(
        cuwisp_io: IO,
        contact_map_distance_limit: float,
        temp_file_directory: str,
        threads_per_block_com_calc: int,
        num_blocks_com_calc: int,
        threads_per_block_sum_coordinates_calc: int,
        num_blocks_sum_coordinates_calc: int,
        num_multiprocessing_processes: int,
        node_coordinate_frames: List[int],
) -> None:
    """
    @param cuwisp_io:
    @type cuwisp_io: IO

    @param contact_map_distance_limit:
    @type contact_map_distance_limit: float

    @param temp_file_directory:
    @type temp_file_directory: str

    @param threads_per_block_com_calc:
    @type threads_per_block_com_calc: int

    @param num_blocks_com_calc:
    @type num_blocks_com_calc: int

    @param threads_per_block_sum_coordinates_calc:
    @type threads_per_block_sum_coordinates_calc: int

    @param num_blocks_sum_coordinates_calc:
    @type num_blocks_sum_coordinates_calc: int

    @param num_multiprocessing_processes:
    @type num_multiprocessing_processes: int

    @param node_coordinate_frames:
    @type node_coordinate_frames: list

    @return:
    @rtype: None

    """
    pdb_single_frame_files = prepare_trajectory_for_analysis(
        temp_file_directory,
        cuwisp_io=cuwisp_io
    )
    average_pdb = Molecule()
    average_pdb_trajectory = (
        md.load(f'{temp_file_directory}/0.pdb')
    )
    average_pdb.parse(
        average_pdb_trajectory
    )
    num_traj_frames, num_atoms, paths, num_frames = (
        get_parameters_for_multiprocessing_pdb_parser(
            temp_file_directory,
            average_pdb
        )
    )
    if not node_coordinate_frames:
        node_coordinate_frames = [
            frame for frame in range(num_traj_frames)
        ]
    pdbs = multiprocessing_pdb_parser(
        num_traj_frames,
        num_atoms,
        num_multiprocessing_processes,
        paths,
        pdb_single_frame_files,
        num_frames,
        average_pdb,
        num_blocks_sum_coordinates_calc,
        threads_per_block_sum_coordinates_calc,
    )
    shutil.rmtree(temp_file_directory)
    average_pdb = pdbs[-1]
    coms = calculate_center_of_masses(
        pdbs,
        threads_per_block_com_calc,
        num_blocks_com_calc,
    )
    for i, pdb in enumerate(pdbs):
        pdb.map_nodes_to_node_tags(coms[i])
    node_atom_indices = (
        serialize_nodes(
            average_pdb=average_pdb,
            coms=coms,
            cuwisp_io=cuwisp_io,
            node_coordinate_frames=node_coordinate_frames,
        )
    )

    num_nodes = len(node_atom_indices)
    num_pdbs = len(pdbs)
    nodes = get_node_com_coordinates_array(
        num_nodes,
        num_pdbs,
        pdbs,
    )
    correlation_matrix = (
        _get_correlation_matrix(
            num_nodes,
            nodes.astype(np.float64),
            average_pdb,
        )
    )
    save_matrix(
        cuwisp_io.correlation_matrix_fname,
        correlation_matrix,
    )

    contact_map, correlation_matrix = get_contact_map(
        correlation_matrix,
        average_pdb,
        contact_map_distance_limit,
        node_atom_indices,
    )

    save_matrix(
        cuwisp_io.contact_map_correlation_matrix_fname,
        correlation_matrix,
    )

    save_matrix(
        cuwisp_io.contact_map_matrix_fname,
        contact_map,
    )
    cuwisp_io.serialize(
        xml_filename=cuwisp_io.io_fname
    )
