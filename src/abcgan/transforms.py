"""
Transforms to and from z-scaled variables.

Uses numpy only (no pytorch)
"""
import numpy as np  # noqa
import abcgan.constants as const


def encode(data, name):
    """
    Encode variables, or just add extra dimension

    Parameters
    ----------
    data : np.ndarray
        array of variable values.
    name : str
        name of the variable.

    Returns
    -------
    enc : np.ndarray
        array of encoded variables (with an extra dimension in all cases)
    """
    if name in const.cyclic_driver:
        wrap_val = const.cyclic_driver[name]
        enc = data % wrap_val
        enc = (enc / wrap_val) * 2.0 * np.pi
        enc = np.stack((np.cos(enc), np.sin(enc)), axis=-1)
    else:
        enc = data[..., None]  # add trailing singleton dimension

    return enc


def decode(data):
    """
    Encode variables, or just add extra dimension

    Parameters
    ----------
    data : np.ndarray
        array of feature values.

    Returns
    -------
    enc : np.ndarray
        array of encoded variables
    """
    curr = 0
    decs = []
    for name in const.driver_names:
        if name in const.cyclic_driver:
            wrap_val = const.cyclic_driver[name]
            cval = data[:, curr]
            sval = data[:, curr + 1]
            theta = np.arctan2(sval, cval)
            dec = theta * wrap_val / 2.0 / np.pi
            dec = dec % wrap_val
            decs.append(dec)
            curr += 2
        else:
            decs.append(data[:, curr])
            curr += 1
    return np.stack(decs, axis=-1)


def compute_valid(bvs):
    # valid only if value within thresholds and if at least one non-zero value
    valid_mask = np.zeros((bvs.shape[0], const.n_bv_feat))
    for i in range(const.n_bv_feat):
        valid_mask[:, i] = ~(((const.bv_thresholds[i][0] > bvs[:, :, i]) |
                              (bvs[:, :, i] > const.bv_thresholds[i][1])).any(-1) |
                             ((bvs[:, :, i] == 0).all(-1)))
    # valid only if every altitude is valid (skip first for now)
    valid_mask = valid_mask.all(-1)
    return valid_mask


def scale_driver(drivers):
    """
    Return a scaled version of the drivers.

    Parameters
    --------------
    drivers: np.ndarray
        n_samples x n_driver

    Returns
    --------------
    driver_feat: np.ndarray
        n_samples x n_driver_feat
    """
    drivers = np.hstack([
        encode(drivers[:, i], n)
        for i, n in enumerate(const.driver_names)
    ])
    drivers = np.log(1 + drivers)
    driver_feat = (drivers - const.driver_mu) / const.driver_sigma
    return driver_feat


def scale_bv(bvs):
    """
    Return a scaled version of the drivers.

    Parameters
    --------------
    bvs: np.ndarray
        n_samples x n_bv

    Returns
    --------------
    bv_feat: np.ndarray
        n_samples x n_bv_feat
    """
    valid_mask = compute_valid(bvs)
    bvs = np.log(1 + bvs)
    bv_feat = (bvs - const.bv_mu) / const.bv_sigma
    # pad bvs to max_alt if needed
    if bv_feat.shape[1] < const.max_alt:
        pad_alt = const.max_alt - bv_feat.shape[1]
        bv_feat = np.pad(bv_feat,
                         ((0, 0), (0, pad_alt), (0, 0)),
                         constant_values=np.nan)
    elif bv_feat.shape[1] > const.max_alt:
        bv_feat = bv_feat[:, :const.max_alt, :]
    return bv_feat, valid_mask


def get_driver(driver_feat):
    """
    Invert featurization to recover driving parameters.

    Parameters
    --------------
    drivers: np.ndarray
        n_samples x n_driver

    Returns
    --------------
    scaled_feat: np.ndarray
        n_samples x n_driver_feat
    """
    drivers = const.driver_sigma * driver_feat + const.driver_mu
    drivers = np.exp(drivers) - 1.0
    drivers = decode(drivers)
    return drivers


def get_bv(bv_feat):
    """
    Invert featurization to recover bvs.

    Parameters
    --------------
    bvs: np.ndarray
        n_samples x n_bv

    Returns
    --------------
    scaled_feat: np.ndarray
        n_samples x n_bv_feat
    """
    bvs = const.bv_sigma * bv_feat + const.bv_mu
    bvs = np.exp(bvs) - 1.0
    return bvs
