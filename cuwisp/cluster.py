from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import os
from collections import defaultdict
from typing import (
    Union,
    Optional,
    List,
)
import numpy as np
from abserdes import Serializer as serializer
from scipy import interpolate
from scipy.spatial import distance
from sklearn.cluster import (
    AffinityPropagation,
)

from .cfrechet import cFrechet
from .splines import Spline

def get_area_between_paths(
        spline_a: Spline,
        spline_b: Spline,
        lower_bound: float,
        upper_bound: float,
) -> np.float64:
    """
    @param spline_a:
    @type: Spline

    @param spline_b:
    @type: Spline

    @param lower_bound:
    @type: float

    @param upper_bound:
    @type: float

    @return:
    @rtype: numpy.float64

    """
    spline_a_integral = spline_a.integrate(
        lower_bound=lower_bound,
        upper_bound=upper_bound
    )
    spline_b_integral = spline_b.integrate(
        lower_bound=lower_bound,
        upper_bound=upper_bound
    )
    area_vector = np.zeros(3, dtype=np.float64)
    area_vector[0] = spline_a_integral[0] - spline_b_integral[0]
    area_vector[1] = spline_a_integral[1] - spline_b_integral[1]
    area_vector[2] = spline_a_integral[2] - spline_b_integral[2]
    area_vector_norm = np.linalg.norm(area_vector)
    return area_vector_norm

def get_area_between_paths_matrix(
        path_splines: List[interpolate.BSpline],
        degree: Optional[int] = 3,
        incr: Optional[float] = 0.001,
        lower_bound: Optional[float] = 0.0,
        upper_bound: Optional[float] = False,
) -> np.ndarray:
    """
    @param path_splines:
    @type: list

    @param degree:
    @type: int, optional

    @param incr:
    @type: float, optional

    @param lower_bound:
    @type: float, optional

    @param upper_bound:
    @type: float, optional

    @return:
    @rtype: numpy.ndarray

    """
    num_paths = len(path_splines)
    if not upper_bound:
        upper_bound = 1.0 + incr
    area_matrix = np.zeros(
        (num_paths, num_paths),
        dtype=np.float64
    )
    for i in range(num_paths):
        for j in range(num_paths):
            area_matrix[i, j] = get_area_between_paths(
                spline_a=path_splines[i],
                spline_b=path_splines[j],
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
    return area_matrix

def get_manhattan_distance_matrix(
        A: np.ndarray
) -> np.ndarray:
    """
    @param A:
    @type: numpy.ndarray

    @return:
    @rtype: numpy.ndarray

    """
    n = min(A.shape)
    MDM = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            MDM[a, b] = -1 * distance.cityblock(
                u=A[a],
                v=A[b],
            )
    return MDM

def cluster_paths(
        A: np.ndarray,
) -> List:
    """
    @param A:
    @type: numpy.ndarray

    @return:
    @rtype: list

    """
    A_MDM = get_manhattan_distance_matrix(A)
    clusters = defaultdict(list)
    affinity_propagation = AffinityPropagation(
        affinity='precomputed'
    )
    affinity_propagation.fit(A_MDM)
    for i in range(len(A)):
        clusters[affinity_propagation.labels_[i]].append(i)
    return clusters

def frechet_distance(
        p: np.ndarray,
        q: np.ndarray,
) -> np.float64:
    """
    @param p:
    @type p: numpy.ndarray

    @param q:
    @type q: numpy.ndarray

    @return:
    @rtype: numpy.float64

    """
    p_size = p.shape[0]
    q_size = q.shape[0]
    P = p.T.reshape(3 * p_size)
    Q = q.T.reshape(3 * q_size)
    ca = np.ones(
        p_size * q_size,
        dtype=np.float64
    ) * -1
    return (
        cFrechet(
            P,
            Q,
            ca,
            p_size - 1,
            q_size - 1,
        )
    )
