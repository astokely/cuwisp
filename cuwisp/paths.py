from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import time
from collections import (
    deque,
    defaultdict,
)
from numba import cuda
import numpy as np
import os
import cupy as cp
import inspect
from typing import (
    Any,
    Tuple,
    Optional,
    List,
    Union,
    Set,
    Deque,
    Callable,
    Dict,
)
from math import floor
from .cuwispio import (
    IO,
    naming_conventions
)
from abserdes import Serializer as serializer
from .nodes import (
    Nodes,
    Node,
)
from .numba_cuda.hedetniemi import hedetniemi_distance

class Rule(object):

    def __init__(
            self,
            append_method: Callable,
            pop_method: Callable,
    ) -> None:
        """
        @param append_method:
        @type append_method: callable

        @param pop_method:
        @type pop_method: callable

        @return:
        @rtype: None

        """
        self._pop = pop_method
        self._append = append_method

    def __repr__(
            self
    ):
        return (
                f'{inspect.getsource(self._append)}\n'
                + f'{inspect.getsource(self._pop)}'
        )

    def pop(
            self,
            dq: Deque,
    ) -> Any:
        """
        @param dq:
        @type dq: deque

        @return:
        @rtype: object

        """
        return self._pop(dq)

    def append(
            self,
            dq: Deque,
            val: Any,
    ) -> None:
        """
        @param dq:
        @type dq: deque

        @param val:
        @type val: object

        @return:
        @rtype: object

        """
        return self._append(dq, val)

class SuboptimalPaths(serializer):

    def __init__(
            self,
            paths: Optional[list] = False,
            src: Optional[int] = None,
            sink: Optional[int] = None,
            num_paths: Optional[int] = None,
    ) -> None:
        """
        @param paths:
        @type paths: list, optional

        @param src:
        @type src: int, optional

        @param sink:
        @type sink: int, optional

        @param num_paths:
        @type num_paths: int, optional

        @return:
        @rtype: None

        """
        if not paths:
            paths = []
        self.paths = paths
        self.src = src
        self.sink = sink
        self.num_paths = num_paths
        return

    def __iter__(
            self
    ):
        for path in self.paths:
            yield path

    def __repr__(
            self
    ):
        suboptimal_paths = ''
        for path in self.paths:
            path = f'{path}\n'
            suboptimal_paths += path
        return suboptimal_paths

    def __getitem__(
            self,
            index
    ):
        if index < len(self.paths):
            return self.paths[index]

    def __setitem__(
            self,
            index,
            path
    ):
        if index >= len(self.paths):
            self.paths.append(path)
        else:
            self.paths[index] = path

    def __len__(
            self
    ):
        return len(self.paths)

    def reverse(
            self
    ):
        """
        @return:
        @rtype: Path

        """
        for path in reversed(self.paths):
            yield path

    def find_paths_with_node(
            self,
            node_index: int
    ) -> List:
        """
        @param node_index:
        @type node_index: int

        @return:
        @rtype: list

        """
        path_indices_with_node = []
        for path in self.paths:
            path_node_indices = [
                node_index for node_indices in [
                    (
                        edge.node1.index,
                        edge.node2.index,
                    ) for edge in path.edges
                ]
                for node_index in node_indices
            ]
            if node_index in path_node_indices:
                path_indices_with_node.append(
                    path
                )
        return path_indices_with_node

    def update(
            self,
            paths: List,
            cutoff: Optional[np.float64] = np.inf,
            remove: Optional[bool] = False,
    ) -> None:
        """
        @param paths:
        @type paths: list

        @param cutoff:
        @type cutoff: numpy.float64, optional

        @param remove:
        @type remove: bool, optional

        @return:
        @rtype: None

        """
        if remove:
            for path in paths:
                self.paths.remove(path)
        else:
            self.paths = self.paths + paths
        paths_dict = {
            path.length: path for path in self.paths
            if path.length <= cutoff
        }
        self.paths = list(
            dict(
                sorted(
                    paths_dict.items()
                )
            ).values()
        )
        self.num_paths = len(self.paths)
        path_index = 0
        for path in self.paths:
            path.index = path_index
            path_index += 1

    def factory(
            self,
            path_indices: List,
    ) -> Any:
        """
        @param path_indices:
        @type path_indices: list

        @return:
        @rtype: object

        """
        path_indices = path_indices
        suboptimal_paths = SuboptimalPaths()
        for index in path_indices:
            suboptimal_paths.paths.append(
                self.paths[index]
            )
        suboptimal_paths.src = self.src
        suboptimal_paths.sink = self.sink
        suboptimal_paths.num_paths = len(
            suboptimal_paths.paths
        )
        return suboptimal_paths

class Path(serializer):

    def __init__(
            self,
            index: Optional[int] = None,
            num_nodes: Optional[int] = None,
            num_edges: Optional[int] = None,
            src: Optional[int] = None,
            sink: Optional[int] = None,
            edges: Optional[list] = None,
            length: Optional[float] = None,
            serialized_splines: Optional[Dict[int, str]] = False,
    ) -> None:
        """
        @param index:
        @type index: int, optional

        @param num_nodes:
        @type num_edges: int, optional

        @param num_edges:
        @type num_nodes: int, optional

        @param src:
        @type src: int, optional

        @param sink:
        @type sink: int, optional

        @param edges:
        @type edges: list, optional

        @param length:
        @type length: float, optional

        @param serialized_splines:
        @type: str, optional

        @return:
        @rtype: None

        """
        if edges is None:
            edges = []
        if not serialized_splines:
            serialized_splines = {}
        self.index = index
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.src = src
        self.sink = sink
        self.edges = edges
        self.length = length
        self.serialized_splines = serialized_splines

    def __repr__(
            self
    ):
        edges = self.edges
        length = self.length
        return f'{edges}: {length}'

    def __iter__(
            self
    ):
        for edge in self.edges:
            yield edge

    def __getitem__(
            self,
            index
    ):
        if index < len(self.edges):
            return self.edges[index]

    def __setitem__(
            self,
            index,
            edge
    ):
        if index >= len(self.edges):
            self.edges.append(edge)
        else:
            self.edges[index] = edge

    def __len__(
            self
    ):
        return len(self.edges)

    def node_coordinates(
            self,
            frame: Optional[int] = 0,
    ) -> np.ndarray:
        node_coordinates = np.zeros(
            shape=(self.num_nodes, 3),
            dtype=np.float64,
        )
        node_indices = [
            edge.node2.index for edge in self.edges
        ]
        node_indices.insert(0, self.edges[0].node1.index)
        all_node_coordinates = np.load(
            f'{self.edges[0].node1.coordinates_directory}'
            + f'/{frame}.npy'
        )
        for i in range(self.num_nodes):
            node_coordinates[i, 0] = (
                all_node_coordinates[node_indices[i], 0]
            )
            node_coordinates[i, 1] = (
                all_node_coordinates[node_indices[i], 1]
            )
            node_coordinates[i, 2] = (
                all_node_coordinates[node_indices[i], 2]
            )
        return node_coordinates

    def resname_count(
            self,
            reverse: Optional[bool] = True,
    ) -> Dict:
        """
        @param reverse:
        @type reverse: bool, optional

        @return:
        @rtype: dict

        """
        resname_count_dict = defaultdict(int)
        for edge in self.edges:
            resname_count_dict[
                edge.node1.resname
            ] += 1
            resname_count_dict[
                edge.node2.resname
            ] += 1
        return {
            resname: count for resname, count in sorted(
                resname_count_dict.items(),
                key=lambda
                    items: items[1],
                reverse=reverse
            )
        }

    @property
    def swap_src_sink(
            self
    ) -> Any:
        """
        @return:
        @rtype: object

        """
        rev_edges = []
        src = self.sink
        sink = self.src
        for edge in reversed(self.edges):
            rev_edge = Edge()
            rev_edge.node1 = edge.node2
            rev_edge.node2 = edge.node1
            rev_edges.append(rev_edge)
        self.edges = rev_edges
        self.src = src
        self.sink = sink
        return self

    def get_edge_index_from_node_index(
            self,
            node_index: int,
            node_edge_pos: int,
    ) -> int:
        """
        @param node_index:
        @type node_index: int

        @param node_edge_pos:
        @type node_edge_pos: int

        @return:
        @rtype: int

        """
        edge_index = 0
        edge_index_dict = {}
        for edge in self.edges:
            if node_edge_pos == 0:
                edge_index_dict[
                    edge.node1.index
                ] = edge_index
            else:
                edge_index_dict[
                    edge.node2.index
                ] = edge_index
            edge_index += 1
        return edge_index_dict[node_index]

    def get_common_src_sink(
            self,
            path: Any,
    ) -> int:
        """
        @param path:
        @type path: Path

        @return:
        @rtype: int

        """
        if self.src == path.src:
            return 0
        elif self.sink == path.sink:
            return 1
        return 2

    def factory(
            self,
            path: Any,
            correlation_matrix: np.ndarray,
    ) -> Any:
        """
        @param path:
        @type path: Path

        @param correlation_matrix:
        @type correlation_matrix: numpy.ndarray

        @return:
        @rtype: Path

        """
        new_path = Path()
        new_path.src = self.src
        new_path.sink = self.sink
        common_src_sink = self.get_common_src_sink(
            path
        )
        if common_src_sink == 2:
            path = path.swap_src_sink
            common_src_sink = self.get_common_src_sink(
                path
            )
        if common_src_sink == 0:
            for edge in path.edges:
                new_path.edges.append(edge)
            new_path.edges = (
                    new_path.edges
                    + self.edges[slice(
                *(
                    self.get_edge_index_from_node_index(
                        path.sink,
                        common_src_sink,
                    ),
                    self.num_edges
                )
            )]
            )
        else:
            new_path.edges = self.edges[slice(
                *(
                    0,
                    self.get_edge_index_from_node_index(
                        path.src,
                        common_src_sink,
                    ) + 1
                )
            )]
            for edge in path.edges:
                new_path.edges.append(edge)
        new_path.num_edges = len(new_path.edges)
        new_path.num_nodes = new_path.num_edges + 1
        new_path.length = np.float64(0.0)
        for edge in new_path.edges:
            new_path.length += correlation_matrix[
                edge.node1.index,
                edge.node2.index,
            ]
        new_path.index = 0
        return new_path

class Edge(serializer):

    def __init__(
            self,
            node1: Optional[Node] = False,
            node2: Optional[Node] = False,
    ) -> None:
        """
        @param node1:
        @type node1: Node, optional

        @param node2:
        @type node2: Node, optional

        @return:
        @rtype: None

        """
        if not node1:
            node1 = Node()
        if not node2:
            node2 = Node()
        self.node1 = node1
        self.node2 = node2

    def __repr__(
            self
    ) -> str:
        return str(
            (
                self.node1.index,
                self.node2.index
            )
        )

    def __iter__(
            self
    ):
        nodes = [self.node1, self.node2]
        for node in nodes:
            yield node

    def __getitem__(
            self,
            index
    ):
        if index == 0:
            return self.node1
        elif index == 1:
            return self.node2

    def __setitem__(
            self,
            index,
            node
    ):
        if index == 0:
            self.node1 = node
        elif index == 1:
            self.node2 = node

    def __len__(
            self
    ):
        return 2

def ordered_paths(
        paths: List,
        src: int,
) -> List:
    """
    @param paths:
    @type paths: list

    @param src:
    @type src: int

    @return:
    @rtype: list

    """
    ordered_paths_list = []
    paths = list(paths)
    pos = None
    while paths:
        for i in reversed(paths):
            if src in i:
                ordered_paths_list.append(i)
                pos = i[1]
                paths.remove(i)
            elif i[0] == pos:
                ordered_paths_list.append(i)
                pos = i[1]
                paths.remove(i)
    return ordered_paths_list

def append_middle(
        dq: deque,
        val: Tuple,
) -> None:
    """
    @param dq:
    @type dq: deque

    @param val:
    @type val: tuple

    @return:
    @rtype: None

    """
    middle_index = floor(len(dq) / 2)
    dq.insert(middle_index, val)

def pop_middle(
        dq: deque,
) -> Tuple:
    """
    @param dq:
    @type dq: deque

    @return:
    @rtype: tuple

    """
    middle_index = floor(len(dq) / 2)
    middle_val = dq[middle_index]
    del dq[middle_index]
    return middle_val

def append(
        dq: deque,
        val: Tuple,
) -> None:
    """

    @param dq:
    @type dq: deque

    @param val:
    @type val: tuple

    @return:
    @rtype: None

    """
    dq.append(val)
    return

def pop(
        dq: deque,
) -> Tuple:
    """
    @param dq:
    @type dq: deque

    @return:
    @rtype: tuple

    """
    return dq.pop()

def append_left(
        dq: deque,
        val: Tuple,
) -> None:
    """
    @param dq:
    @type dq: deque

    @param val:
    @type val: tuple

    @return:
    @rtype: None

    """
    dq.appendleft(val)
    return

def pop_left(
        dq: deque,
) -> Tuple:
    """
    @param dq:
    @type dq: deque

    @return:
    @rtype: tuple

    """
    return dq.popleft()

def built_in_rules() -> Dict[int, Rule]:
    """
    @return:
    @rtype: dict

    """
    return {
        0: Rule(append, pop),
        1: Rule(append_middle, pop_middle),
        2: Rule(append, pop_left),
        3: Rule(append_middle, pop_left),
        4: Rule(append_middle, pop),
    }

def get_ssp(
        src: int,
        sink: int,
        h: np.ndarray,
        a: np.ndarray,
) -> Tuple:
    """
    @param src:
    @type src: int

    @param sink:
    @type sink: int

    @param h:
    @type h: numpy.ndarray

    @param a:
    @type a: numpy.ndarray

    @return:
    @rtype: tuple

    """
    pos = sink
    p = h[src][sink]
    h_row = h[:, src]
    a_col = a[sink, :]
    path = []
    nodes = [pos]
    while pos != src:
        h_row_sorted_indices = cp.argsort(
            h_row
        )
        closest = {}
        found_next_node = False
        for i in h_row_sorted_indices:
            if h_row[i] == np.inf:
                break
            dist = h_row[i] + a_col[i]
            if i != pos:
                if p == dist == np.inf:
                    return
                closest[abs(dist - p)] = i
            if dist == p and i != pos:
                prev_pos = pos
                pos = i
                a_col = a[pos, :]
                path.append((pos, prev_pos))
                nodes.append(pos)
                p = h_row[pos]
                found_next_node = True
                break
        if not found_next_node:
            minn = min(closest.keys())
            prev_pos = pos
            pos = closest[minn]
            a_col = a[pos, :]
            path.append((pos, prev_pos))
            nodes.append(pos)
            p = h_row[pos]
    return path, nodes

def serialize_correlation_matrix(
        a: np.ndarray,
        serialization_fname: str,
        round_index: int,
        correlation_matrix_serialization_directory: str,
        naming_convention: Callable,
) -> None:
    """
    @param a:
    @type a: numpy.ndarray

    @param serialization_fname:
    @type serialization_fname: str

    @param round_index:
    @type round_index: int

    @param correlation_matrix_serialization_directory:
    @type correlation_matrix_serialization_directory: str

    @param naming_convention:
    @type: callable

    @return:
    @rtype: None

    """
    numpy_fname = (
        f'{correlation_matrix_serialization_directory}/'
        + naming_convention(
            'serialized',
            serialization_fname,
            'correlation',
            'matrix',
            str(round_index),
        ) + f'.npy'
    )
    np.save(numpy_fname, a)

def serialize_suboptimal_paths(
        src: int,
        sink: int,
        serialization_fname: str,
        ssp: np.ndarray,
        nodes: Nodes,
        s: Set,
        round_index: int,
        suboptimal_paths_serialization_directory: str,
        naming_convention: Callable,
) -> None:
    """
    @param src:
    @type src: int

    @param sink:
    @type sink: int

    @param serialization_fname:
    @type serialization_fname: str

    @param ssp:
    @type ssp: numpy.ndarray

    @param nodes:
    @type nodes: Nodes

    @param s:
    @type s: set

    @param round_index:
    @type round_index: int

    @param suboptimal_paths_serialization_directory:
    @type suboptimal_paths_serialization_directory: str

    @param naming_convention:
    @type: callable

    @return:
    @rtype: None

    """
    d = {i[-1]: i[:-1] for i in s}
    d[ssp[-1]] = ssp[:-1]
    path_index = 0
    suboptimal_paths = SuboptimalPaths()
    for k in sorted(d):
        path = Path()
        path.length = k
        path.edges = []
        path_nodes = set([])
        for path_edge in ordered_paths(d[k], src):
            node1_index, node2_index = path_edge
            path_nodes.add(node1_index)
            path_nodes.add(node2_index)
            edge = Edge()
            edge.node1 = nodes[node1_index]
            edge.node2 = nodes[node2_index]
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
    xml_fname = (
        f'{suboptimal_paths_serialization_directory}/'
        + naming_convention(
            'serialized',
            serialization_fname,
            'suboptimal',
            'paths',
            str(round_index),
        ) + f'.xml'
    )
    suboptimal_paths.serialize(xml_fname)

def explore_paths(
        src: int,
        sink: int,
        a: np.ndarray,
        nodes: List,
        n: int,
        cutoff: Union[float, None],
        threads_per_block: int,
        serialization_fname: str,
        serialization_frequency: int,
        nodes_obj: Nodes,
        ssp: np.ndarray,
        round_index: int,
        correlation_matrix_serialization_directory: str,
        suboptimal_paths_serialization_directory: str,
        max_num_paths: int,
        rule: Rule,
        naming_convention: Callable,
) -> Set:
    """
    @param src:
    @type src: int

    @param sink:
    @type sink: int

    @param a:
    @type a: numpy.ndarray

    @param nodes:
    @type nodes: list

    @param n:
    @type n: int

    @param cutoff:
    @type cutoff: float, None

    @param threads_per_block:
    @type threads_per_block: int

    @param serialization_fname:
    @type serialization_fname: str

    @param serialization_frequency:
    @type serialization_frequency: int

    @param nodes_obj:
    @type nodes_obj: Nodes

    @param ssp:
    @type ssp: numpy.ndarray

    @param round_index:
    @type round_index: int

    @param correlation_matrix_serialization_directory:
    @type correlation_matrix_serialization_directory: str

    @param suboptimal_paths_serialization_directory:
    @type suboptimal_paths_serialization_directory: str

    @param max_num_paths:
    @type max_num_paths: int

    @param rule:
    @type rule: Rule

    @param naming_convention
    @type: callable

    @return:
    @rtype: set

    """
    if serialization_frequency:
        serialize_correlation_matrix(
            a,
            serialization_fname,
            round_index,
            correlation_matrix_serialization_directory,
            naming_convention,
        )
    q = deque([])
    s = set([])
    n_p = set([])
    for i in nodes:
        q.append(i)
        n_p.add(i)
    start = time.time()
    while q:
        if len(s) > max_num_paths:
            break
        for i in nodes:
            n_p.add(i)
        i, j = rule.pop(q)
        a[i][j] = np.inf
        a[j][i] = np.inf
        h = np.array(
            hedetniemi_distance(
                a,
                n,
                threads_per_block,
                cutoff
            )
        )
        if get_ssp(src, sink, h, a) is not None:
            path, nodes = get_ssp(src, sink, h, a)
        else:
            break
        path.append(h[src][sink])
        prev_s_size = len(s)
        s.add(tuple(path))
        new_s_size = len(s)
        if new_s_size != prev_s_size:
            if serialization_frequency:
                if (time.time() - start) > serialization_frequency:
                    serialize_correlation_matrix(
                        a,
                        serialization_fname,
                        round_index,
                        correlation_matrix_serialization_directory,
                        naming_convention,
                    )
                    serialize_suboptimal_paths(
                        src,
                        sink,
                        serialization_fname,
                        ssp,
                        nodes_obj,
                        s,
                        round_index,
                        suboptimal_paths_serialization_directory,
                        naming_convention,
                    )
                    start = time.time()
        nodes = [
            (nodes[i], nodes[i + 1])
            for i in range(len(nodes) - 1)
        ]
        for i in nodes:
            if i not in n_p:
                rule.append(q, i)
        if not q:
            break
    return s

def get_suboptimal_paths(
        cuwisp_io: IO,
        src: int,
        sink: int,
        cutoff: Union[float, None],
        threads_per_block: int,
        serialization_frequency: int,
        correlation_matrix_serialization_directory: str,
        suboptimal_paths_serialization_directory: str,
        simulation_rounds: List[int],
        gpu_index: int,
        max_num_paths: int,
        rules: Dict,
        correlation_matrix_fname: str,
) -> None:
    """
    @param cuwisp_io
    @type: IO

    @param src:
    @type src: int

    @param sink:
    @type sink: int

    @param cutoff:
    @type cutoff: float, None

    @param threads_per_block:
    @type threads_per_block: int

    @param serialization_frequency:
    @type serialization_frequency: int

    @param correlation_matrix_serialization_directory:
    @type correlation_matrix_serialization_directory: str

    @param suboptimal_paths_serialization_directory:
    @type suboptimal_paths_serialization_directory: str

    @param simulation_rounds:
    @type simulation_rounds: list

    @param gpu_index:
    @type gpu_index: int

    @param max_num_paths:
    @type max_num_paths: int

    @param rules:
    @type rules: dict

    @param correlation_matrix_fname:
    @type: str

    @return:
    @rtype: None

    """
    naming_convention = naming_conventions()[
        cuwisp_io.naming_convention
    ]
    ssp = None
    cuda.select_device(gpu_index)
    suboptimal_paths_dict = {}
    nodes_obj = Nodes()
    nodes_obj.deserialize(cuwisp_io.nodes_fname)
    for simulation_round in simulation_rounds:
        a = np.array(
            np.load(
                correlation_matrix_fname
            )
        )
        n = len(a)
        h = np.array(
            hedetniemi_distance(
                a,
                n,
                threads_per_block,
                np.inf,
            )
        )
        if get_ssp(src, sink, h, a) is None:
            raise Exception(
                "Sink node is unreachable from source node.".upper()
                + '\n'
                + "Either perform the suboptimal path calculation" +
                '\n'
                + "using the correlation matrix without the contact "
                  "map" + '\n'
                + "applied, or rerun the correlation matrix "
                  "calculation with" + '\n'
                + "a larger cutoff distance."
            )

        if not os.path.exists(
            cuwisp_io.all_pairs_shortest_paths_matrix_fname
        ):
            np.save(
                cuwisp_io.all_pairs_shortest_paths_matrix_fname,
                h,
            )
        path, nodes = get_ssp(src, sink, h, a)
        ssp = path
        ssp.append(h[src][sink])
        if not cutoff:
            cutoff = ssp[-1] * np.float64(1.2)
        nodes = [(nodes[i], nodes[i + 1]) for i in
            range(len(nodes) - 1)]
        paths = list(
            explore_paths(
                src,
                sink,
                a,
                nodes,
                n,
                cutoff,
                threads_per_block,
                cuwisp_io.calc_name,
                serialization_frequency,
                nodes_obj,
                ssp,
                simulation_round,
                correlation_matrix_serialization_directory,
                suboptimal_paths_serialization_directory,
                max_num_paths,
                rules[simulation_round],
                naming_convention,
            )
        )
        for path in paths:
            suboptimal_paths_dict[path[-1]] = path[:-1]
    suboptimal_paths_dict[ssp[-1]] = ssp[:-1]
    path_index = 0
    suboptimal_paths = SuboptimalPaths()
    for path_length in sorted(suboptimal_paths_dict):
        path = Path()
        path.length = path_length
        path.edges = []
        path_nodes = set([])
        for path_edge in ordered_paths(
                suboptimal_paths_dict[path_length], src
        ):
            node1_index, node2_index = path_edge
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
    suboptimal_paths.serialize(
        cuwisp_io.suboptimal_paths_fnames[-1]
    )
