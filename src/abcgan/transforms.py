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


def decode(data, driver_names):
    """
    Encode variables, or just add extra dimension

    Parameters
    ----------
    data : np.ndarray
        array of feature values.
    driver_names: list: str
        list driver names in data
    Returns
    -------
    enc : np.ndarray
        array of encoded variables
    """
    curr = 0
    decs = []
    for name in driver_names:
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


def compute_valid(bvs, bv_thresholds=const.bv_thresholds):
    """
    Returns a mask which can be used to get rid of
    invalid background variables samples and outliers

    Parameters
    --------------
    bvs: np.ndarray
        (n_samples x n_waves x n_hfp_feat)
    bv_thresholds: np.ndarray:
        Upper and lower bounds of each bv feature.
    Returns
    --------------
    valid_mask: bool np.ndarray
        (n_samples,)
    """
    # valid only if value within thresholds and if at least one non-zero value
    valid_mask = np.zeros((bvs.shape[0], bv_thresholds.shape[0])) > 0
    for i in range(bv_thresholds.shape[0]):
        valid_mask[:, i] = ~(((bvs[:, :, i] < bv_thresholds[i][0]) |
                             (bvs[:, :, i] > bv_thresholds[i][1])).any(-1) |
                             ((np.isnan(bvs[:, :, i])).all(-1)) |
                             (np.isnan(bvs[:, 0, i])))
    # valid only if every altitude is valid
    valid_mask = valid_mask.all(-1)
    return valid_mask


def compute_valid_hfp(hfps, hfp_thresholds=const.hfp_thresholds):
    """
    Returns a mask which can be used to get rid of
    invalid hfp waves and outliers

    Parameters
    --------------
    hfps: np.ndarray
        (n_samples x n_waves x n_hfp_feat)
    hfp_thresholds: np.ndarray:
        Upper and lower bounds of each hfp feature.
    Returns
    --------------
    valid_mask: bool np.ndarray
        (n_samples,)
    """

    # valid only if value within thresholds
    valid_mask = np.zeros((hfps.shape[0], hfp_thresholds.shape[0])) > 0

    for i in range(hfps.shape[1]):
        valid_mask[:, i] = ~(((hfps[:, :, i] < hfp_thresholds[i][0]) |
                             (hfps[:, :, i] > hfp_thresholds[i][1])).any(-1))
    # valid only if every altitude is valid
    valid_mask = valid_mask.any(-1)
    return valid_mask


def scale_driver(drivers, driver_names=const.driver_names):
    """
    Return a scaled version of the drivers.

    Parameters
    --------------
    drivers: np.ndarray
        n_samples x n_driver
    driver_names: list: str
        list of driver names
    Returns
    --------------
    driver_feat: np.ndarray
        n_samples x n_driver_feat
    """
    driver_feat = np.hstack([
        encode(drivers[:, i], n)
        for i, n in enumerate(driver_names[:drivers.shape[1]])
    ])
    dr_idxs = np.hstack([const.dr_feat_map[name] for name in driver_names])
    log_idxs = [i for (i, d) in enumerate(dr_idxs) if d in const.log_drs]

    driver_feat[:, log_idxs] = np.log(1 + driver_feat[:, log_idxs])
    driver_feat = (driver_feat - const.driver_mu[dr_idxs]) / const.driver_sigma[dr_idxs]
    return driver_feat


def scale_hfp(hfps):
    """
    Return a scaled version of the hfps.

    Parameters
    --------------
    hfps: np.ndarray
        n_samples x n_waves x n_hfp

    Returns
    --------------
    hfp_feat: np.ndarray
        n_samples x n_waves x n_hfp_feat
    """
    # Don't compute valid mask if no bvs are available
    if hfps.shape[1] > 0:
        valid_mask = compute_valid_hfp(hfps)
    else:
        valid_mask = np.ones(hfps.shape[0], dtype=bool)

    hfp_feat = hfps.copy()
    hfp_feat[:, :, const.invert_hfp] *= -1
    hfp_feat[:, :, const.log_hfp] = np.log(1 + hfp_feat[:, :, const.log_hfp])
    hfp_feat = (hfp_feat - const.hfp_mu) / const.hfp_sigma
    return hfp_feat, valid_mask


def get_hfp(hfp_feat):
    """
    Invert featurization to recover hfp.

    Parameters
    --------------
    hfp_feat: np.ndarray
        n_samples x n_hfp_feat

    Returns
    --------------
    hfps: np.ndarray
        n_samples x n_hfp
    """
    hfps = const.hfp_sigma * hfp_feat + const.hfp_mu
    hfps[:, :, const.log_hfp] = np.exp(hfps[:, :, const.log_hfp]) - 1.0
    hfps[:, :, const.invert_hfp] *= -1
    return hfps


def scale_bv(bvs, bv_type='radar'):
    """
    Return a scaled version of the drivers.

    Parameters
    --------------
    bvs: np.ndarray
        n_samples x n_bv
    bv_type: str
        string specifying weather to scale

    Returns
    --------------
    bv_feat: np.ndarray
        n_samples x n_bv_feat
    """
    if bv_type == 'lidar':
        thresholds = const.lidar_thresholds
        bv_mu = const.lidar_bv_mu
        bv_sigma = const.lidar_bv_sigma
        max_alt = const.max_alt_lidar
    else:
        thresholds = const.bv_thresholds
        bv_mu = const.bv_mu
        bv_sigma = const.bv_sigma
        max_alt = const.max_alt

    # Don't compute valid mask if no bvs are available
    if bvs.shape[1] > 0:
        valid_mask = compute_valid(bvs, thresholds)
    else:
        valid_mask = np.ones(bvs.shape[0], dtype=bool)

    if bv_type == 'lidar':
        bv_feat = np.log(1 + bvs)
    else:
        bv_feat = bvs.copy()
        bv_feat[:, :, const.log_bvs] = np.log(1 + bv_feat[:, :,  const.log_bvs])
    bv_feat = (bv_feat - bv_mu) / bv_sigma
    # pad bvs to max_alt if needed
    if bv_feat.shape[1] < max_alt:
        pad_alt = max_alt - bv_feat.shape[1]
        bv_feat = np.pad(bv_feat,
                         ((0, 0), (0, pad_alt), (0, 0)),
                         constant_values=np.nan)
    elif bv_feat.shape[1] > max_alt:
        bv_feat = bv_feat[:, :max_alt, :]
    return bv_feat, valid_mask


def get_driver(driver_feat, driver_names=const.driver_names):
    """
    Invert featurization to recover driving parameters.

    Parameters
    --------------
    driver_feat: np.ndarray
        n_samples x n_driver_feat
    driver_names: list: str
        list driver names in driver_feat
    Returns
    --------------
    original driver: np.ndarray
        n_samples x n_driver
    """
    dr_idxs = np.hstack([const.dr_feat_map[name] for name in driver_names])
    log_idxs = [i for (i, d) in enumerate(dr_idxs) if d in const.log_drs]

    drivers = const.driver_sigma[dr_idxs] * driver_feat + const.driver_mu[dr_idxs]
    drivers[:, log_idxs] = np.exp(drivers[:, log_idxs]) - 1.0
    drivers = decode(drivers, driver_names)
    return drivers


def get_bv(bv_feat, bv_type='radar'):
    """
    Invert featurization to recover bvs.

    Parameters
    --------------
    bv_feat: np.ndarray
        n_samples x n_bv_feat
    bv_type: str
        radar or lidar bvs

    Returns
    --------------
    scaled_feat: np.ndarray
        n_samples x n_bv
    """
    if bv_type == 'lidar':
        bvs = const.lidar_bv_sigma * bv_feat + const.lidar_bv_mu
        bvs = np.exp(bvs) - 1.0
    else:
        bvs = const.bv_sigma * bv_feat + const.bv_mu
        bvs[:, :, const.log_bvs] = np.exp(bvs[:, :, const.log_bvs]) - 1.0
    return bvs
