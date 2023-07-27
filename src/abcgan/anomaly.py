import numpy as np
import abcgan.transforms as trans
from scipy.special import logsumexp
from numpy import linalg as LA
import abcgan.constants as const


def marginal_anomaly_estimation(sampled_feats: np.ndarray, feats: np.ndarray, alpha: float = 2.0):
    """
        compute an anomaly scores from the set of generated features
        using marginal distribution based logsumexp computation

        Parameters
        -------------
        sampled_feats: np.ndarray
            sampled z-scaled features for each input feature
        feats: np.ndarray
            input feature broadcast to match sampled_feat dim
        alpha: float
            scalar parameter for sigma (lower alpha --> finner resolution)
        Returns
        -------------
        anomalies: np.ndarray
             marginal anomaly scores (feat.shape)
        """

    std = np.std(sampled_feats, axis=0)[None, ...]
    sigma = std / len(sampled_feats) * alpha

    a = -0.5 * ((sampled_feats - feats) / sigma) ** 2
    b = 1 / (sigma * np.sqrt(2 * np.pi))
    anomalies = logsumexp(a=a, b=b, axis=0)
    return anomalies


def joint_anomaly_estimation(sampled_feats: np.ndarray, feats: np.ndarray, alpha: float = 1.0):
    """
        compute an anomaly scores from the set of generated features
        using joint distribution based logsumexp computation

        Parameters
        -------------
        sampled_feats: np.ndarray
            sampled z-scaled features for each input feature
        feats: np.ndarray
            input feature broadcast to match sampled_feat dim
        alpha: float
            scalar parameter for sigma (lower alpha --> finner resolution)
        Returns
        -------------
        anomalies: np.ndarray
            joint anomaly scores (feat.shape[:-1])
        """
    (n, n_feats) = sampled_feats.shape[0], sampled_feats.shape[-1]

    sigma = (np.prod(np.std(sampled_feats, axis=0), axis=-1) / n) ** (1 / n_feats) * alpha
    a = -0.5 * LA.norm(sampled_feats - feats, axis=-1) ** 2 / sigma ** 2
    b = 1 / (sigma * np.sqrt(2 * np.pi))
    anomalies = logsumexp(a=a, b=b, axis=0)

    return anomalies


def anomaly_score_bv(bvs, gen_bvs, method: str = 'joint', alpha: float = 2.0):
    """
    returns unbounded anomaly scores for a background profiles
    given set of generated samples. Low scores represent anomalous events.

    Parameters
    -------------
    bvs: np.ndarray
         (n_samples x n_alt x n_bv) real background profiles
    gen_bvs: np.ndarray
        (n_samples x n_repeat x n_alt x n_feat) n_repeat generated
         set of generated background profiles for each input sample
    method: str, optional
        'joint': estimates a single anomaly score for each altitude bin using joint distribution
        'marginal': estimates anomaly scores at each alt for each bv feature using marginal distributions
    alpha: float
        scalar parameter for sigma (lower alpha --> finner resolution)
    Returns
    -------------
    anomalies: np.ndarray
        (n_samples x n_alt)  output of anomaly scores if joint
        (n_samples x n_alt x n_feat) output of anomaly scores if marginal

    """
    if method == 'joint':
        anomalies = np.zeros(bvs.shape[:-1])
    elif method == 'marginal':
        anomalies = np.zeros_like(bvs)
    else:
        raise ValueError(f"{method} is an invalid method."
                         f" Pleas use 'marginal' or 'joint' method")
    n_alts = bvs.shape[-2]
    for i in range(len(bvs)):
        if bvs.shape[-1] == const.n_bv_feat:
            G_feats = trans.scale_bv(gen_bvs[i, ...])[0][:, :n_alts]
            real_feat = trans.scale_bv(bvs[[i], ...])[0][:, :n_alts]
        elif bvs.shape[-1] == const.n_lidar_bv_feat:
            G_feats = trans.scale_bv(gen_bvs[i, ...], bv_type='lidar')[0][:, :n_alts]
            real_feat = trans.scale_bv(bvs[[i], ...], bv_type='lidar')[0][:, :n_alts]
        else:
            raise ValueError(f"{bvs.shape[-1]} is an invalid number of input features. "
                             f"Must input {const.n_bv} for Radar BVs or "
                             f"{const.n_lidar_bv} for Lidar BVs.")
        if method == 'joint':
            anomalies[i, :] = joint_anomaly_estimation(G_feats, real_feat, alpha=alpha)
        else:
            anomalies[i, :, :] = marginal_anomaly_estimation(G_feats, real_feat, alpha=alpha)

    return anomalies


def anomaly_score_hfp(hfps: np.ndarray, gen_hfps: np.ndarray, method: str = 'joint', alpha: float = 2.0):
    """
    returns unbounded anomaly scores for a HFP waves
    given set of generated samples. Low scores represent anomalous events.

    Parameters
    -------------
    hfps: np.ndarray
         (n_samples x n_waves x n_bv) real HFPs
    gen_hfps: np.ndarray
        (n_samples x n_repeat x n_waves x n_feat) n_repeat generated
         set of generated hfp waves for each input sample
    method: str, optional
        'joint': estimates a single anomaly score for each wave using joint distribution
        'marginal': estimates anomaly scores on each hfp feature for each wave using marginal distributions
    alpha: float
        scalar parameter for sigma (lower alpha --> finner resolution)
    Returns
    -------------
    anomalies: np.ndarray
        (n_samples x n_waves)  output of anomaly scores if joint
        (n_samples x n_waves x n_feat) output of anomaly scores if marginal

    """
    if method == 'joint':
        anomalies = np.zeros(hfps.shape[:-1])
    elif method == 'marginal':
        anomalies = np.zeros_like(hfps)
    else:
        raise ValueError(f"{method} is an invalid method."
                         f" Pleas use 'marginal' or 'joint' method")

    for i in range(len(hfps)):
        if hfps.shape[-1] == const.n_hfp_feat:
            G_feats = trans.scale_hfp(gen_hfps[i, ...])[0]
            real_feat = trans.scale_hfp(hfps[[i], ...])[0]
        else:
            raise ValueError(f"{hfps.shape[-1]} is an invalid number of input features. "
                             f"Must input {const.n_hfp} for HFP Waves ")
        if method == 'joint':
            anomalies[i, :] = joint_anomaly_estimation(G_feats, real_feat, alpha=alpha)
        else:
            anomalies[i, :, :] = marginal_anomaly_estimation(G_feats, real_feat, alpha=alpha)

    return anomalies


def anomaly_score_wtec(wtecs: np.ndarray,
                       gen_wtecs: np.ndarray,
                       method: str = 'joint',
                       alpha: float = 2.0,
                       tid_type: str = const.wtec_default_tid_type):
    """
    returns unbounded anomaly scores for TEC wave parameters
    given set of generated TEC waves. Low scores represent anomalous events.

    Parameters
    -------------
    wtecs: np.ndarray
         (n_samples x n_wtec) real background profiles
    gen_wtecs: np.ndarray
        (n_samples x n_repeat x n_wtec) n_repeat generated
         set of generated tec waves for each input sample
    method: str, optional
        'marginal': estimates anomaly scores on each tec feature using marginal distributions
        'joint': estimates anomaly score of each tec wave using joint distribution
    alpha: float
        scalar parameter for sigma (lower alpha --> finner resolution)
    tid_type: str
        specify dataset type for z-scaling
    Returns
    -------------
    anomalies: np.ndarray
        (n_samples)  output of anomaly scores if joint
        (n_samples x n_feat) output of anomaly scores if marginal

    """
    if method == 'joint':
        anomalies = np.zeros(wtecs.shape[:-1])
    elif method == 'marginal':
        anomalies = np.zeros_like(wtecs)
    else:
        raise ValueError(f"{method} is an invalid method."
                         f" Pleas use 'marginal' or 'joint' method")

    for i in range(len(wtecs)):
        if wtecs.shape[-1] == const.n_wtec:
            G_feats = trans.scale_wtec(gen_wtecs[i, ...], tid_type=tid_type)[0]
            real_feat = trans.scale_wtec(wtecs[[i], ...], tid_type=tid_type)[0]
        else:
            raise ValueError(f"{wtecs.shape[-1]} is an invalid number of input features. "
                             f"Must input {const.n_wtec} for TEC Waves ")
        if method == 'joint':
            anomalies[i, ...] = joint_anomaly_estimation(G_feats, real_feat, alpha=alpha)
        else:
            anomalies[i, ...] = marginal_anomaly_estimation(G_feats, real_feat, alpha=alpha)

    return anomalies
