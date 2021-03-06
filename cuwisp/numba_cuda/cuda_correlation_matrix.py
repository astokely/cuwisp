from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import numpy as np
from math import (
    fabs,
    sqrt,
    log,
    ceil,
)
from numba import (
    float64,
    cuda,
)

def cuda_correlation_matrix(
        nodes: np.ndarray,
        average_nodes: np.ndarray,
        correlation_matrix: np.ndarray
):
    """
    @param nodes:
    @type nodes: numpy.ndarray

    @param average_nodes:
    @type average_nodes: numpy.ndarray

    @param correlation_matrix:
    @type correlation_matrix: numpy.ndarray

    @return:
    @rtype: numpy.ndarray

    """
    tpb = 32

    @cuda.jit
    def cuda_node_coordinate_deviations(
            cu_nodes,
            cu_average_node,
            node_coordinate_deviations
    ):
        """
        @param cu_nodes:
        @type cu_nodes: numpy.ndarray

        @param cu_average_node:
        @type cu_average_node: numpy.ndarray

        @param node_coordinate_deviations:
        @type node_coordinate_deviations: numpy.ndarray

        @return:
        @rtype: None

        """
        # noinspection PyArgumentList
        i, j = cuda.grid(2)
        if i < cu_nodes.shape[0] and j < cu_nodes.shape[1]:
            node_coordinate_deviations[i][j][0] = (
                    cu_nodes[i][j][0] - cu_average_node[i][0]
            )
            node_coordinate_deviations[i][j][1] = (
                    cu_nodes[i][j][1] - cu_average_node[i][1]
            )
            node_coordinate_deviations[i][j][2] = (
                    cu_nodes[i][j][2] - cu_average_node[i][2]
            )
        return

    num_nodes = nodes.shape[0]
    num_pdbs = nodes.shape[1]
    h_node_coordinate_deviations = np.zeros(
        (num_nodes, num_pdbs, 3),
        dtype=np.float64
    )

    h_ensemble_average_delta_square_magnitudes = np.zeros(
        num_nodes,
        dtype=np.float64
    )
    d_ensemble_average_delta_square_magnitudes = cuda.to_device(
        h_ensemble_average_delta_square_magnitudes
    )

    d_nodes = cuda.to_device(nodes)
    d_average_nodes = cuda.to_device(average_nodes)
    d_node_coordinate_deviations = cuda.to_device(
        h_node_coordinate_deviations
    )

    threadsperblock = (32, 32)
    blockspergrid = (
        ceil(nodes.shape[0] / threadsperblock[0]),
        ceil(nodes.shape[1] / threadsperblock[1]),
    )
    cuda_node_coordinate_deviations[blockspergrid, threadsperblock](
        d_nodes,
        d_average_nodes,
        d_node_coordinate_deviations
    )
    d_node_coordinate_deviations.copy_to_host(
        h_node_coordinate_deviations
    )
    d_ensemble_average_delta_square_magnitudes.copy_to_host(
        h_ensemble_average_delta_square_magnitudes
    )
    tpb = 256

    @cuda.jit
    def cuSqMag(
            a,
            b,
            c,
    ):
        """
        @param a:
        @type a: numpy.ndarray

        @param b:
        @type b: numpy.ndarray

        @param c:
        @type c: numpy.ndarray

        @return:
        @rtype: None

        """
        bid = cuda.blockIdx.x
        y = bid
        while y < a.shape[0]:
            sm = cuda.shared.array(
                (tpb, 3),
                float64
            )
            tid = cuda.threadIdx.x
            bdim = cuda.blockDim.x
            lid = tid
            sm[lid][0] = 0.0
            sm[lid][1] = 0.0
            sm[lid][2] = 0.0
            cuda.syncthreads()
            while lid < a.shape[1]:
                sm[tid][0] += (
                        a[y][lid][0] * b[y][lid][0]
                )
                sm[tid][1] += (
                        a[y][lid][1] * b[y][lid][1]
                )
                sm[tid][2] += (
                        a[y][lid][2] * b[y][lid][2]
                )
                lid += bdim
            cuda.syncthreads()
            sweep = tpb // 2
            while sweep > 0:
                if tid < sweep:
                    sm[tid][0] += sm[tid + sweep][0]
                    sm[tid][1] += sm[tid + sweep][1]
                    sm[tid][2] += sm[tid + sweep][2]
                cuda.syncthreads()
                sweep = sweep // 2
            if tid == 0:
                c[y] = (sm[0][0] + sm[0][1] + sm[0][2]) / a.shape[1]
            y += cuda.gridDim.x

    threadsperblock = tpb
    blockspergrid = (
        ceil(num_nodes / threadsperblock),
    )
    cuSqMag[blockspergrid, threadsperblock](
        d_node_coordinate_deviations,
        d_node_coordinate_deviations,
        d_ensemble_average_delta_square_magnitudes,
    )
    d_ensemble_average_delta_square_magnitudes.copy_to_host(
        h_ensemble_average_delta_square_magnitudes
    )
    tpb = 32

    @cuda.jit
    def correlation_matrix_kernel(
            node_coordinate_deviations,
            ensemble_average_delta_square_magnitudes,
            cu_correlation_matrix,
    ):
        """
        @param node_coordinate_deviations:
        @type node_coordinate_deviations: numpy.ndarray

        @param ensemble_average_delta_square_magnitudes:
        @type ensemble_average_delta_square_magnitudes: numpy.ndarray

        @param cu_correlation_matrix:
        @type cu_correlation_matrix: numpy.ndarray

        @return:
        @rtype: None

        """
        # noinspection PyArgumentList
        i, j = cuda.grid(2)
        if (
                i < node_coordinate_deviations.shape[0]
                and j < node_coordinate_deviations.shape[0]
        ):
            x = 0.0
            y = 0.0
            z = 0.0
            for n in range(node_coordinate_deviations.shape[1]):
                x += (
                        node_coordinate_deviations[i, n, 0]
                        * node_coordinate_deviations[j, n, 0]
                )
                y += (
                        node_coordinate_deviations[i, n, 1]
                        * node_coordinate_deviations[j, n, 1]
                )
                z += (
                        node_coordinate_deviations[i, n, 2]
                        * node_coordinate_deviations[j, n, 2]
                )
            tmp = fabs(
                (x + y + z)
                / node_coordinate_deviations.shape[1]
            )
            cu_correlation_matrix[i, j] = -1 * log(
                tmp / sqrt(
                    ensemble_average_delta_square_magnitudes[i]
                    * ensemble_average_delta_square_magnitudes[j]
                )
            )
        return

    d_correlation_matrix = cuda.to_device(correlation_matrix)
    threadsperblock = (32, 32)
    blockspergrid_x = ceil(
        max(h_node_coordinate_deviations.shape)
        / threadsperblock[0]
    )
    blockspergrid_y = ceil(
        max(h_node_coordinate_deviations.shape)
        / threadsperblock[0]
    )
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    correlation_matrix_kernel[blockspergrid, threadsperblock](
        d_node_coordinate_deviations,
        d_ensemble_average_delta_square_magnitudes,
        d_correlation_matrix,
    )
    d_correlation_matrix.copy_to_host(correlation_matrix)
    return correlation_matrix
