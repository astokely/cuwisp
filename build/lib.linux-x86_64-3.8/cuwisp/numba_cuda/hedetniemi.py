from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

from numba import cuda
import numpy as np
import math
from typing import Tuple

def cuda_grid_and_block_dims(
        num_nodes: int,
        threads_per_block: int,
) -> Tuple:
    dimBlock = (threads_per_block, threads_per_block)
    dimGrid = (
        math.ceil(num_nodes / threads_per_block),
        math.ceil(num_nodes / threads_per_block)
    )
    return dimGrid, dimBlock

@cuda.jit
def init_matrix(
        a,
        b,
        h,
        num_nodes,
):
    i, j = cuda.grid(2)
    if i < num_nodes and j < num_nodes:
        h[i, j] = np.inf
        b[i, j] = a[i, j]

@cuda.jit
def all_pair_hedetniemit(
        a,
        b,
        h,
        num_nodes,
        found_ssp,
        cutoff,
):
    i, j = cuda.grid(2)
    if i < num_nodes and j < num_nodes:
        hedetniemit_sum = np.inf
        for a_row_b_col_index in range(num_nodes):
            hedetniemit_sum = min(
                hedetniemit_sum,
                b[i, a_row_b_col_index] + a[a_row_b_col_index, j]
            )
        h[i, j] = hedetniemit_sum
        if h[i, j] >= cutoff:
            h[i, j] = np.inf
        if b[i, j] != h[i, j]:
            b[i, j] = h[i, j]
            found_ssp[0] = False

def hedetniemi_distance(
        a: np.ndarray,
        num_nodes: int,
        threads_per_block: int,
        cutoff: np.float64,
) -> np.ndarray:
    threads_per_block = int(np.sqrt(threads_per_block))
    n = num_nodes
    a_device = cuda.to_device(a)
    b_device = cuda.device_array(shape=(n, n))
    h_device = cuda.device_array(shape=(n, n))
    dimGrid, dimBlock = (
        cuda_grid_and_block_dims(n, threads_per_block)
    )
    init_matrix[dimGrid, dimBlock](
        a_device,
        b_device,
        h_device,
        num_nodes
    )
    for i in range(n):
        found_ssp = cuda.to_device([True])
        dimGrid, dimBlock = (
            cuda_grid_and_block_dims(n, threads_per_block)
        )
        all_pair_hedetniemit[dimGrid, dimBlock](
            a_device,
            b_device,
            h_device,
            num_nodes,
            found_ssp,
            cutoff,
        )
        if found_ssp[0]:
            break
    h_host = h_device.copy_to_host()
    return h_host
