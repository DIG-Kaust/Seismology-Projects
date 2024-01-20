from math import floor

import numpy as np
from numba import jit, prange

from pylops import LinearOperator
from pylops.utils.decorators import reshaped


@jit(nopython=True, fastmath=True, nogil=True, parallel=True)
def nmo_forward(data, taxis, haxis, vels_rms):
    dt = taxis[1] - taxis[0]
    ot = taxis[0]
    nt = len(taxis)
    nh = len(haxis)

    dnmo = np.zeros_like(data)

    # Parallel outer loop on slow axis
    for ih in prange(nh):
        h = haxis[ih]
        for it0, (t0, vrms) in enumerate(zip(taxis, vels_rms)):
            # Compute NMO traveltime
            tx = np.sqrt(t0**2 + (h / vrms) ** 2)
            it_frac = (tx - ot) / dt  # Fractional index
            it_floor = floor(it_frac)
            it_ceil = it_floor + 1
            w = it_frac - it_floor
            if 0 <= it_floor and it_ceil < nt:  # it_floor and it_ceil must be valid
                # Linear interpolation
                dnmo[ih, it0] += (1 - w) * data[ih, it_floor] + w * data[ih, it_ceil]
    return dnmo


@jit(nopython=True, fastmath=True, nogil=True, parallel=True)
def nmo_adjoint(dnmo, taxis, haxis, vels_rms):
    dt = taxis[1] - taxis[0]
    ot = taxis[0]
    nt = len(taxis)
    nh = len(haxis)

    data = np.zeros_like(dnmo)

    # Parallel outer loop on slow axis; use range if Numba is not installed
    for ih in prange(nh):
        h = haxis[ih]
        for it0, (t0, vrms) in enumerate(zip(taxis, vels_rms)):
            # Compute NMO traveltime
            tx = np.sqrt(t0**2 + (h / vrms) ** 2)
            it_frac = (tx - ot) / dt  # Fractional index
            it_floor = floor(it_frac)
            it_ceil = it_floor + 1
            w = it_frac - it_floor
            if 0 <= it_floor and it_ceil < nt:
                # Linear interpolation
                # In the adjoint, we must spread the same it0 to both it_floor and
                # it_ceil, since in the forward pass, both of these samples were
                # pushed onto it0
                data[ih, it_floor] += (1 - w) * dnmo[ih, it0]
                data[ih, it_ceil] += w * dnmo[ih, it0]
    return data


class NMO(LinearOperator):
    r"""NMO correction

    2D NMO correction operator to be applied to a dataset of size :math:`n_h \times n_t`.

    Parameters
    ----------
    taxis : :obj:`np.ndarray`
       Time axis
    haxis : :obj:`np.ndarray`
       Spatial axis
    vels_rms : :obj:`np.ndarray`
       Velocity profile over time to be used for NMO correction
    dtype : :obj:`str`, optional
       Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
       Operator shape
    dims : :obj:`tuple`
       Model dimensions
    dimsd : :obj:`tuple`
       Dats dimensions
    explicit : :obj:`bool`
       Operator contains a matrix that can be solved explicitly
       (``True``) or not (``False``)

   """
    def __init__(self, taxis, haxis, vels_rms, dtype=None):
        self.taxis = taxis
        self.haxis = haxis
        self.vels_rms = vels_rms

        dims = (len(haxis), len(taxis))
        if dtype is None:
            dtype = np.result_type(taxis.dtype, haxis.dtype, vels_rms.dtype)
        super().__init__(dims=dims, dimsd=dims, dtype=dtype)

    @reshaped
    def _matvec(self, x):
        return nmo_forward(x, self.taxis, self.haxis, self.vels_rms)

    @reshaped
    def _rmatvec(self, y):
        return nmo_adjoint(y, self.taxis, self.haxis, self.vels_rms)
