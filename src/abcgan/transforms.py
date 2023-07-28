"""
Transforms to and from z-scaled variables.

Uses numpy only (no pytorch)
"""
import numpy as np  # noqa
import abcgan.constants as const
from typing import Union, List


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


def encode(data: np.ndarray, name: str):
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


def decode(data: np.ndarray, driver_names: List[str]):
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


def compute_valid(bvs: np.ndarray,
                  bv_thresholds: np.ndarray = const.bv_thresholds):
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


def compute_valid_hfp(hfps: np.ndarray,
                      hfp_thresholds: np.ndarray = const.hfp_thresholds):
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
                       wtec_thresholds: Union[np.ndarray, None] = None,
                       spike_threshold: int = 250):
    """
    Returns a mask which can be used to get rid of
    invalid tec waves and outliers

    Parameters
    --------------
    wtec: np.ndarray
        (n_samples x n_waves x n_wtec)
    wtec_thresholds: np.ndarray:
        Upper and lower bounds of each wtec variable.
    spike_threshold: np.ndarray:
        Max unique values count filter
    Returns
    --------------
    valid_mask: bool np.ndarray
        (n_samples,)
    """
    if wtec_thresholds is None:
        wtec_thresholds = np.stack((np.ones(wtec.shape[-1]) * -np.inf,
                                    np.ones(wtec.shape[-1]) * np.inf), axis=-1)

    # valid only if value within thresholds
    valid_mask = np.ones((wtec.shape[0], wtec.shape[1]), dtype=bool)

    for i in range(wtec.shape[1]):
        valid_mask[:, i] &= ~(((wtec[..., i] < wtec_thresholds[i, 0]) |
                               (wtec[..., i] > wtec_thresholds[i, 1])))
        unique, unique_counts = np.unique(wtec[:, i], return_counts=True)
        filter_values = unique[unique_counts > spike_threshold]
        for j in range(len(filter_values)):
            valid_mask[:, i] &= ~(wtec[:, i] == filter_values[j])
        valid_mask[:, i] &= np.isfinite(wtec[..., i])
    # valid only if every altitude is valid
    valid_mask = valid_mask.all(-1)
    return valid_mask


def scale_driver(drivers: np.ndarray,
                 driver_names: list = None,
                 data_type: str = 'isr'):
    """
    Return a scaled version of the drivers.

    Parameters
    --------------
    drivers: np.ndarray
        n_samples x n_driver
    driver_names: list: str
        list of driver names
    data_type: list: str
        either 'isr' for bv and hfps or 'wtec' for gnss data
    Returns
    --------------
    driver_feat: np.ndarray
        n_samples x n_driver_feat
    """
    if data_type == 'wtec':
        driver_feat_names = const.wtec_dr_feat_names
        mu = const.wtec_driver_mu
        sigma = const.wtec_driver_sigma
        if driver_names is None:
            driver_names = const.wtec_dr_names
    else:
        driver_feat_names = const.driver_feat_names
        mu = const.driver_mu
        sigma = const.driver_sigma
        if driver_names is None:
            driver_names = const.driver_names
    driver_feat = np.hstack([
        encode(drivers[:, i], n)
        for i, n in enumerate(driver_names[:drivers.shape[1]])
    ])
    dr_idxs = np.hstack([const.dr_feat_map[name] for name in driver_names])
    log_idxs = [i for (i, d) in enumerate(dr_idxs) if driver_feat_names[d] in const.log_driver_feats]

    driver_feat[:, log_idxs] = np.log(1 + driver_feat[:, log_idxs])
    driver_feat = (driver_feat - mu[dr_idxs]) / sigma[dr_idxs]
    return driver_feat


def get_driver(driver_feat: np.ndarray,
               driver_names: Union[List[str], None] = None,
               data_type: str = 'isr'):
    """
    Invert featurization to recover driving parameters.

    Parameters
    --------------
    driver_feat: np.ndarray
        n_samples x n_driver_feat
    driver_names: list: str
        list driver names in driver_feat
    data_type: list: str
        either 'isr' for bv and hfps or 'wtec' for gnss data
    Returns
    --------------
    original driver: np.ndarray
        n_samples x n_driver
    """
    if data_type == 'wtec':
        driver_feat_names = const.wtec_dr_feat_names
        mu = const.wtec_driver_mu
        sigma = const.wtec_driver_sigma
        if driver_names is None:
            driver_names = const.wtec_dr_names
    else:
        driver_feat_names = const.driver_feat_names
        mu = const.driver_mu
        sigma = const.driver_sigma
        if driver_names is None:
            driver_names = const.driver_names

    dr_idxs = np.hstack([const.dr_feat_map[name] for name in driver_names])
    log_idxs = [i for (i, d) in enumerate(dr_idxs) if driver_feat_names[d] in const.log_driver_feats]

    drivers = sigma[dr_idxs] * driver_feat + mu[dr_idxs]
    drivers[:, log_idxs] = np.exp(drivers[:, log_idxs]) - 1.0
    drivers = decode(drivers, driver_names)
    return drivers


def scale_hfp(hfps: np.ndarray):
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


def get_hfp(hfp_feat: np.ndarray):
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


def scale_bv(bvs: np.ndarray, bv_type: str = 'radar'):
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


def get_bv(bv_feat: np.ndarray, bv_type: str = 'radar'):
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
               tid_type: str = 'SSTIDs'):
    """
    Return a scaled version of the tec waves.

    Parameters
    --------------
    wtec: np.ndarray
        n_samples x n_wtec_waves x n_wtec
    tid_type: str
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
        valid_mask = compute_valid_wtec(wtec,
                                        wtec_thresholds=const.wtec_dict[tid_type]["thresholds"],
                                        spike_threshold=1000)
    else:
        valid_mask = np.ones(wtec.shape[0], dtype=bool)

    if tid_type in const.wtec_dict.keys():
        mu = const.wtec_dict[tid_type]['mu']
        std = const.wtec_dict[tid_type]['sigma']
    else:
        raise ValueError(f"{tid_type} is invalid. Plz choose from:"
                         f" {list(const.wtec_dict.keys())}")

    wtec_feat = (wtec - mu) / std
    return wtec_feat, valid_mask


def get_wtec(wtec_feat: np.ndarray,
             tid_type: str = "SSTIDs"):
    """
    Invert featurization to recover tec waves.

    Parameters
    --------------
    wtec_feat: np.ndarray
        n_samples x n_wtec_feat
    tid_type: str
        specify dataset type for z-scaling

    Returns
    --------------
    wtec: np.ndarray
        n_samples x n_wtec
    """
    if tid_type in const.wtec_dict.keys():
        mu = const.wtec_dict[tid_type]['mu']
        std = const.wtec_dict[tid_type]['sigma']
    else:
        raise ValueError(f"{tid_type} is invalid. Plz choose from:"
                         f" {list(const.wtec_dict.keys())}")

    wtec = std * wtec_feat + mu
    return wtec
