from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import os
from typing import (
    Union,
    Optional,
    List,
    Any,
)
import numpy as np
from abserdes import Serializer as serializer
from scipy import interpolate
from .paths import SuboptimalPaths

class Spline(serializer):

    def __init__(
            self,
            t: Optional[np.ndarray] = False,
            c: Optional[np.ndarray] = False,
            k: Optional[int] = 3,
            nd_curve: Optional[np.ndarray] = False,
    ) -> None:
        """
        @param t:
        @type: numpy.ndarray, optional

        @param c:
        @type: numpy.ndarray, optional

        @param k:
        @type: int, optional

        @param nd_curve:
        @type: numpy.ndarray, optional

        @return
        @rtype: None

        """
        self._t = t
        self._c = c
        self.k = k
        self._bspline = False
        self._nd_curve = False
        self.serialization_directory = False

    @property
    def t(
            self
    ) -> np.ndarray:
        """
        @return:
        @rtype: numpy.ndarray

        """
        if isinstance(self._t, str):
            self._t = np.load(self._t)
        return self._t

    @t.setter
    def t(
            self,
            t_: Union[str, np.ndarray]
    ) -> None:
        """
        @param t_:
        @type: str, np.ndarray

        @return
        @rtype: None

        """
        if isinstance(t_, list):
            t_ = np.array(t_, dtype=np.float64)
        self._t = t_

    @property
    def c(
            self
    ) -> np.ndarray:
        """
        @return
        @rtype: object

        """
        if isinstance(self._c, str):
            self._c = np.load(self._c)
        return self._c

    @c.setter
    def c(
            self,
            c_: Union[str, np.ndarray]
    ) -> None:
        """
        @param c_:
        @type: str, np.ndarray

        @return
        @rtype: None

        """
        if isinstance(c_, list):
            c_ = np.array(c_, dtype=np.float64)
        self._c = c_

    @property
    def nd_curve(
            self
    ) -> np.ndarray:
        """

        @return:
        @rtype: numpy.ndarray

        """
        if isinstance(self._nd_curve, str):
            self._nd_curve = np.load(self._nd_curve)
        return self._nd_curve

    # noinspection PyMethodOverriding

    @nd_curve.setter
    def nd_curve(
            self,
            nd_curve_: np.ndarray
    ) -> np.ndarray:
        """
        @param nd_curve_:
        @type: numpy.array

        @return:
        @rtype: None

        """
        self._nd_curve = nd_curve_


    def serialize(
            self,
            fname: Optional[str] = False,
    ) -> None:
        """
        @param: fname
        @type: str, optional

        @return:
        @rtype: None

        """
        t, c = self.t, self.c
        nd_curve = self.nd_curve
        root = os.getcwd()
        if os.getcwd() in fname:
            root = ''
        path = os.path.dirname(
            f'{root}/{fname}'
        )
        self.serialization_directory = path
        if not os.path.exists(path):
            os.makedirs(path)
        if self._bspline:
            self._bspline = False
        if isinstance(self.t, np.ndarray):
            t_fname = f'{fname[:-4]}_t.npy'
            np.save(t_fname, self.t)
            self._t = t_fname
        if isinstance(self.c, np.ndarray):
            c_fname = f'{fname[:-4]}_c.npy'
            np.save(c_fname, self.c)
            self._c = c_fname
        if isinstance(self.nd_curve, np.ndarray):
            nd_curve_fname = (
                f'{fname[:-4]}_nd_curve.npy'
            )
            np.save(nd_curve_fname, self.nd_curve)
            self.nd_curve = nd_curve_fname
        super().serialize(
            xml_filename=fname
        )
        self.t, self.c = t, c
        self.nd_curve = nd_curve

    # noinspection PyMethodOverriding
    def deserialize(
            self,
            fname: str,
    ) -> Any:
        """
        @param fname:
        @type: str

        @return:
        @rtype: Spline

        """
        super().deserialize(root=fname)
        return self

    def tck(
            self,
            a: np.ndarray,
            k: Optional[int] = False
    ) -> List:
        """
        @param a:
        @type: numpy.ndarray

        @param k:
        @type: int, optional

        @return:
        @rtype: list

        """
        if not k:
            k = self.k
        tck, u = interpolate.splprep(
            x=a, s=0, k=k
        )
        self.t, self.c, self.k = tck
        return tck

    def evaluate(
            self,
            a: np.ndarray,
    ) -> np.ndarray:
        """
        @param a:
        @type: numpy.ndarray

        @return:
        @rtype: numpy.ndarray

        """
        if not isinstance(self.t, np.ndarray):
            self.tck(a=a)
        if not self._bspline:
            c = np.array(self.c, dtype=np.float64).T
            self._bspline = interpolate.BSpline(
                self.t, c, self.k
            )
        self.nd_curve = np.array((self._bspline(a)))
        return self.nd_curve

    def integrate(
            self,
            lower_bound: float,
            upper_bound: float,
    ) -> np.ndarray:
        """
        @param lower_bound:
        @type: float

        @param upper_bound:
        @type: float

        @return:
        @rtype: numpy.ndarray

        """
        if not self._bspline:
            c = np.array(self.c, dtype=np.float64).T
            self._bspline = interpolate.BSpline(
                self.t, c, self.k
            )
        return self._bspline.integrate(
            a=lower_bound,
            b=upper_bound,
        )

def generate_splines(
        suboptimal_paths: SuboptimalPaths,
        suboptimal_paths_fname: str,
        frames: Optional[List[int]] = False,
        incr: Optional[float] = 0.001,
        degree: Optional[int] = 3,
        output_directory: Optional[str] = False,
) -> None:
    """
    @param suboptimal_paths:
    @type: SuboptimalPaths

    @param suboptimal_paths_fname:
    @type: str

    @param frames:
    @type: list, optional

    @param incr:
    @type: float, optional

    @param degree:
    @type: int, optional

    @param output_directory:
    @type: str, optional

    @return:
    @rtype: None

    """
    if not frames:
        frames = [0]
    if not output_directory:
        output_directory = f'{os.getcwd()}/splines'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    u = np.arange(0.0, 1.0 + incr, incr)
    paths = {
        frame: suboptimal_paths.paths
        for frame in frames
    }
    for frame in paths:
        frame_directory = f'{output_directory}/frame{frame}'
        os.makedirs(frame_directory)
        spline_index = 0
        for path in paths[frame]:
            spline_fname = (
                f'{frame_directory}/spline{spline_index}.xml'
            )
            spline = Spline(k=degree)
            spline.tck(path.node_coordinates(frame).T)
            spline.evaluate(a=u)
            spline.serialize(spline_fname)
            if not path.serialized_splines:
                path.serialized_splines = {}
            path.serialized_splines[frame] = spline_fname
            spline_index += 1
    suboptimal_paths.serialize(
        xml_filename=suboptimal_paths_fname
    )




