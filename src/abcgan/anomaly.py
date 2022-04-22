import numpy as np
import abcgan.transforms as trans
from scipy.special import logsumexp
from numpy import linalg as LA
import abcgan.constants as const


def anomaly_estimation_1d(fakes, data, alpha=2.0):
    """
        compute an unbounded anomaly score for a new data sample using logsumexp computation method

        Parameters
        -------------
        fakes: torch.Tensor
            n_samples x n_alt or n_wave x n_features background variables
        data: torch.Tensor
            1 x n_alt x n_features broadcast n_samples times to match fakes data shape
        alpha: float
            scalar parameter for sigma (lower alpha --> finner resolution)
        Returns
        -------------
        anomalies: 1 xnp.ndarray, np.ndarray
            n_alt x n_feat output of anomaly scores (unbounded).
        """

    std = np.std(fakes, axis=0)[None, :, :]
    sigma = std / len(fakes) * alpha

    a = -0.5 * ((fakes - data) / sigma) ** 2
    b = 1 / (sigma * np.sqrt(2 * np.pi))
    anomalies = logsumexp(a=a, b=b, axis=0)
    return anomalies


def anomaly_estimation_nd(fakes, data, alpha=1.0):
    """
        compute an unbounded anomaly score for a background profile sample at each
        altitude bin or for an hfp wave using logsumexp computation method (N-dimensional)

        Parameters
        -------------
        fakes: np.ndarray
            n_samples x n_alt or n_wave x n_features background variables (z-scaled)
        data: np.ndarray
            1 x n_alt x n_features broadcast n_samples times to match fakes data shape  (z-scaled)
        alpha: float
            scalar parameter for sigma (lower alpha --> finner resolution)
        Returns
        -------------
        anomalies: np.ndarray
            n_alt x 1
        """
    (n, n_alts, n_feats) = fakes.shape

    sigma = (np.prod(np.std(fakes, axis=0), axis=-1) / n) ** (1 / n_feats) * alpha
    a = -0.5 * LA.norm(fakes - data, axis=2) ** 2 / sigma ** 2
    b = 1 / (sigma * np.sqrt(2 * np.pi))
    anomalies = logsumexp(a=a, b=b, axis=0)

    return anomalies


def anomaly_score(real, gen_samples, nd_est=False, alpha=2.0):
    """
    returns unbounded anomaly scores for a background profiles or
    hfp wave given set of generated samples. The more positive numbers
    are more confident.

    Parameters
    -------------
    real: np.ndarray
         (n_samples x n_alt or n_wave x n_feat) real background profiles or hfps
    gen_samples: np.ndarray
        (n_samples x n_repeat x n_alt or n_wave x n_feat) n_repeat generated
         background profiles or hfp waves for input sample
    nd_est: bool, optional
        set for nd anomaly estimation vs 1d
    alpha: float
        scalar parameter for sigma (lower alpha --> finner resolution)
    Returns
    -------------
    anomalies: np.ndarray
        (n_samples x n_alt or n_wave)  output of anomaly scores if nd_est is True
        (n_samples x n_alt or n_wave x n_feat) output of anomaly scores if nd_est is False

    """
    if nd_est:
        anomalies = np.zeros((len(real), real.shape[1]))
    else:
        anomalies = np.zeros_like(real)

    for i in range(len(real)):
        if real.shape[-1] == const.n_bv_feat:
            G_feats = trans.scale_bv(gen_samples[i, ...])[0]
            real_feat = trans.scale_bv(real[[i], ...])[0]
        elif real.shape[-1] == const.n_lidar_bv_feat:
            G_feats = trans.scale_bv(gen_samples[i, ...], bv_type='lidar')[0]
            real_feat = trans.scale_bv(real[[i], ...], bv_type='lidar')[0]
        elif real.shape[-1] == const.n_hfp_feat:
            G_feats = trans.scale_hfp(gen_samples[i, ...])[0]
            real_feat = trans.scale_hfp(real[[i], ...])[0]
        else:
            raise ValueError(f"{real.shape[-1]} is an invalid number of input features. "
                             f"Must input {const.n_bv_feat} for Radar BVs, "
                             f"{const.n_lidar_bv_feat} for Lidar BVs, or "
                             f"{const.n_hfp_feat} for HFP data.")
        if nd_est:
            anomalies[i, :] = anomaly_estimation_nd(G_feats, real_feat, alpha=alpha)
        else:
            anomalies[i, :, :] = anomaly_estimation_1d(G_feats, real_feat, alpha=alpha)

    return anomalies
