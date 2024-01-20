import numpy as np
from scipy.signal import filtfilt

def semblance(d, x, t, t0, vrms, nsmooth=5):
    """Semblance Stack

    Parameters
    ----------
    d : np.ndarray
        Data (nt x nx)
    x : np.ndarray
        Spatial axis
    t : np.ndarray
        Time axis
    t0 : np.ndarray
        Zero-offset time axis
    vrms : np.ndarray
        Root-mean-square velocity axis
    nsmooth : int, optional
        Length of smoothing filter

    Returns
    -------
    ss : np.ndarray
        Semblance stack (nt0 x nvrms)

    """
    # identify sampling and dimensions
    dt = t[1]-t[0]  # time lenght
    nx = x.size  # number of samples
    nt = t.size  # number of time measurements
    nvrms = vrms.size  # number of velocity estimates in the velocity spectra
    nt0 = t0.size  # number of time-zero measurements in the velocity spectra

    # compute semblance
    ix = np.arange(nx)  # offset axis
    num = np.zeros((nt0, nvrms))  # numerator in semblance formula
    den = np.zeros((nt0, nvrms))  # denominator in semblance formula
    nn = np.zeros((nt0, nvrms))  # length in semblance formula

    num_ab = np.zeros((nt0, nvrms))  # numerator in AB semblance formula
    den_ab = np.zeros((nt0, nvrms))  # denominator in AB semblance formula

    for it0, t0_ in enumerate(t0):
        # iterates over the vrms-t0 spectra
        for ivrms, vrms_ in enumerate(vrms):
            tevent = np.sqrt(t0_ ** 2 + x ** 2 / vrms_ **
                             2)  # traveltime of the event
            # normalizes the hyperbola and places it on the origin
            tevent = (tevent - t[0]) / dt
            itevent = tevent.astype(int)
            dtevent = tevent - itevent
            mask = (itevent < nt - 1) & (itevent >= 0)
            dss = d[itevent[mask], ix[mask]] * \
                (1 - dtevent[mask]) + \
                d[itevent[mask], ix[mask]] * dtevent[mask]
            xss = x[mask]

            # AB semblance
            num_ab[it0, ivrms] = (2 * np.sum(dss) * np.sum(xss) * np.sum(dss * xss)) - \
            (np.sum(dss)**2 * np.sum(xss**2)) - (len(dss) * np.sum(dss*xss)**2)
            den_ab[it0, ivrms] = np.sum(
                dss**2) * (np.sum(xss)**2 - len(dss) * np.sum(xss**2))

            # Ordinary semblance
            num[it0, ivrms] = np.sum(dss)**2
            den[it0, ivrms] = np.sum(dss**2)
            nn[it0, ivrms] = len(dss)

    ss = num / (nn * den + 1e-10)
    ss_ab = num_ab / (den_ab + 1e-10)

    # smooth along time axis
    ss = filtfilt(np.ones(nsmooth) / nsmooth, 1, ss.T).T
    ss_ab = filtfilt(np.ones(nsmooth) / nsmooth, 1, ss_ab.T).T

    return ss, ss_ab


def nmocorrection(d, x, t0, vrms, vmask=None):
    """Normal moveout correction
    
    Parameters
    ----------
    d : np.ndarray
        Data (nt x nx)
    x : np.ndarray
        Spatial axis
    t0 : np.ndarray
        Zero-offset time axis
    vrms : np.ndarray
        Root-mean-square velocity axis
    vmask : int, optional
        Velocity of linear event used as mask
    
    Returns
    -------
    dnmo : np.ndarray
        NMO corrected data stack (nt x nx)

    """
    # identify sampling and dimensions
    dt = t0[1]-t0[0]
    nx = x.size
    nvrms = vrms.size
    nt0 = t0.size
    
    # flatten events
    ix = np.arange(nx)
    dnmo = np.zeros((nt0, nx))
    for it0, t0_ in enumerate(t0):
        tevent = np.sqrt(t0_ ** 2 + x ** 2 / vrms[it0] ** 2)
        tevent = (tevent - t[0]) / dt
        itevent = tevent.astype(int)
        dtevent = tevent - itevent
        mask = (itevent < nt - 2) & (itevent >= 0)
        nmask = np.sum(mask)
        dnmo[it0 * np.ones(nmask, dtype='int'), ix[mask]] += \
            d[itevent[mask], ix[mask]] * (1 - dtevent[mask]) + d[itevent[mask] + 1, ix[mask]] * dtevent[mask]
    
    # apply mask
    mask = np.ones((nt0, nx))
    if vmask:
        tmask = x / vmask
        itmask = (tmask / dt).astype(int)
        for ix in range(nx):
            mask[:itmask[ix], ix] = 0.
        dnmo *= mask
    fold = np.sum(mask, axis=1)
    return dnmo, fold