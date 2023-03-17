"""
Transforms to and from z-scaled variables.

Uses numpy only (no pytorch)
"""
import numpy as np  # noqa
import abcgan.constants as const
from typing import Union


def encoded_driver_names(dr_names: list = const.driver_names):
    """
    Gets list of driver feature names in order of
    Parameters
    ----------
    dr_names : list
        array of driver names.

    Returns
    -------
    driver_feat_names : list
        list of driver feature names
    """
    driver_feat_names = []
    for dn in dr_names:
        if dn in const.cyclic_driver:
            driver_feat_names.append('cos_' + dn)
            driver_feat_names.append('sin_' + dn)
        else:
            driver_feat_names.append(dn)
    return driver_feat_names


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


def compute_valid_wtec(wtec: np.ndarray,
                       wtec_thresholds: np.ndarray = const.wtec_thresholds):
    """
    Returns a mask which can be used to get rid of
    invalid tec waves and outliers

    Parameters
    --------------
    wtec: np.ndarray
        (n_samples x n_waves x n_wtec)
    wtec_thresholds: np.ndarray:
        Upper and lower bounds of each wtec variable.
    Returns
    --------------
    valid_mask: bool np.ndarray
        (n_samples,)
    """

    # valid only if value within thresholds
    valid_mask = np.zeros((wtec.shape[0], wtec.shape[1])) > 0

    for i in range(wtec.shape[1]):
        valid_mask[:, i] = ~(((wtec[..., i] < wtec_thresholds[i, 0]) |
                             (wtec[..., i] > wtec_thresholds[i, 1])).any(-1))
    # valid only if every altitude is valid
    valid_mask = valid_mask.any(-1)
    return valid_mask


def scale_driver(drivers: np.ndarray,
                 driver_names: list = const.driver_names):
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
    log_idxs = [i for (i, d) in enumerate(dr_idxs) if const.driver_feat_names[d] in const.log_driver_feats]

    driver_feat[:, log_idxs] = np.log(1 + driver_feat[:, log_idxs])
    driver_feat = (driver_feat - const.driver_mu[dr_idxs]) / const.driver_sigma[dr_idxs]
    return driver_feat



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
    log_idxs = [i for (i, d) in enumerate(dr_idxs) if const.driver_feat_names[d] in const.log_driver_feats]

    drivers = const.driver_sigma[dr_idxs] * driver_feat + const.driver_mu[dr_idxs]
    drivers[:, log_idxs] = np.exp(drivers[:, log_idxs]) - 1.0
    drivers = decode(drivers, driver_names)
    return drivers


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


def scale_wtec(wtec: np.ndarray,
               dataset_name: Union[None, str] = const.wtec_default_dataset):
    """
    Return a scaled version of the tec waves.

    Parameters
    --------------
    wtec: np.ndarray
        n_samples x n_wtec_waves x n_wtec
    dataset_name: str
        specify dataset type for z-scaling

    Returns
    --------------
    wtec_feat: np.ndarray
        n_samples x n_wtec_waves x n_wtec_feat
    valid_mask: np.ndarray
        n_samples x 1
    """
    # Don't compute valid mask if no bvs are available
    if wtec.shape[1] > 0:
        valid_mask = compute_valid_wtec(wtec)
    else:
        valid_mask = np.ones(wtec.shape[0], dtype=bool)

    wtec_feat = wtec.copy()
    wtec_feat[:, const.log_wtec] = np.log(1 + wtec_feat[:, const.log_wtec])

    if dataset_name is None:
        mu = const.wtec_zscale_dict[const.wtec_default_dataset]['mu']
        std = const.wtec_zscale_dict[const.wtec_default_dataset]['sigma']
    elif dataset_name in const.wtec_datasets_names:
        mu = const.wtec_zscale_dict[dataset_name]['mu']
        std = const.wtec_zscale_dict[dataset_name]['sigma']
    else:
        raise ValueError(f"{dataset_name} is invalid. Plz choose from: {const.wtec_datasets_names}")

    wtec_feat = (wtec_feat - mu) / std
    return wtec_feat, valid_mask


def get_wtec(wtec_feat: np.ndarray,
             dataset_name: Union[None, str] = const.wtec_default_dataset):
    """
    Invert featurization to recover tec waves.

    Parameters
    --------------
    wtec_feat: np.ndarray
        n_samples x n_wtec_feat
    dataset_name: str
        specify dataset type for z-scaling

    Returns
    --------------
    wtec: np.ndarray
        n_samples x n_wtec
    """
    if dataset_name is None:
        mu = const.wtec_zscale_dict[const.wtec_default_dataset]['mu']
        std = const.wtec_zscale_dict[const.wtec_default_dataset]['sigma']
    elif dataset_name in const.wtec_datasets_names:
        mu = const.wtec_zscale_dict[dataset_name]['mu']
        std = const.wtec_zscale_dict[dataset_name]['sigma']
    else:
        raise ValueError(f"{dataset_name} is invalid. Plz choose from: {const.wtec_datasets_names}")

    wtec = std * wtec_feat + mu
    wtec[:, const.log_wtec] = np.exp(wtec[:, const.log_wtec]) - 1.0
    return wtec
