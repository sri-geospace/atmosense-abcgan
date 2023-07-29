import torch
import numpy as np
from typing import Union, Tuple
from scipy import stats

import abcgan.constants as const
import abcgan.transforms as trans


def get_clusters(drivers: np.ndarray, n_clusters: int = 10):
    """
    Returns cluster indicating conditional subsets based on driver values

    Parameters
    ----------------
    drivers: np.ndarray
        array of drivers
    n_clusters: int
        number of clusters subsets to create for each driver feature

    Returns
    -------------
    clusters:
        array of clusters for each sample and input driver feat
    """
    clusters = np.zeros_like(drivers, dtype=int)
    for d in range(drivers.shape[-1]):
        steps = np.ones(n_clusters, dtype=int) * len(drivers) // n_clusters
        for i in range(len(drivers) % n_clusters):
            steps[i] += 1
        idxs = np.hstack((np.array([0]), np.cumsum(steps)))
        driver_sort_idxs = np.argsort(drivers[:, d])
        for i in range(1, len(idxs)):
            clusters[driver_sort_idxs[idxs[i - 1]: idxs[i]], d] = i - 1
    return clusters.astype(int)


def conditional_wtec_scores(drivers: np.ndarray,
                            real_wtecs: np.ndarray,
                            generated_wtecs: np.ndarray,
                            n_clusters: int = 5,
                            tid_type=const.wtec_default_tid_type,
                            return_hist_info: bool = False,
                            **hellinger_kwargs):
    """
    Returns the hellinger distance scores across various conditional subsets derived
    from the geo physical driver values

    Parameters
    ----------------
    drivers: np.ndarray
        drivers to condition subsets with (Not Z-Scaled)
    real_wtecs: np.ndarray
        array of tec wave parameters (Not Z-Scaled)
    generated_wtecs: np.ndarray
        array of generated tec wave parameters (Not Z-Scaled)
    n_clusters: int
        number of conditional subsets to create for each driver feature
    tid_type: str
       TID type for scaling
    return_hist_info:
        set to return histograms and edges
    hellinger_kwargs: dict
        hellinger distance calculation arguments

    Returns
    -------------
    cond_dist:
        the hellinger distance for each cluster, driver and wtec feature
         (n_drivers or n_clusters x n_wtec)
    hist_info:
        additional histogram data that was used to calculate hellinger dist.
        Info includes to real and fake histograms and there shared bin edges.
    """
    if n_clusters < 1:
        raise ValueError(f'n_clusters = {n_clusters} is invalid. Must be greater than zero.')

    if hellinger_kwargs.get("z_scale") is not None:
        hellinger_kwargs.pop("z_scale")
    if hellinger_kwargs.get("z_scale_inputs") is not None:
        hellinger_kwargs.pop("z_scale_inputs")
    clusters = get_clusters(drivers, n_clusters=n_clusters)
    cond_dist = np.zeros((drivers.shape[-1], n_clusters, real_wtecs.shape[-1]))
    hist_info = [[] for i in range(drivers.shape[-1])]

    for i in range(drivers.shape[-1]):
        for j in range(n_clusters):
            c_mask = clusters[:, i] == j
            cond_dist[i, j, :], info = hellinger_scores_wtec(real_wtecs[c_mask],
                                                             generated_wtecs[c_mask],
                                                             z_scale=False, z_scale_inputs=False,
                                                             tid_type=tid_type,
                                                             return_hist_info=True,
                                                             **hellinger_kwargs)
            hist_info[i].append(info)
    if return_hist_info:
        return cond_dist, hist_info
    else:
        return cond_dist


def hellinger_score_kde(ds1: np.ndarray,
                        ds2: np.ndarray,
                        binranges: Union[None, tuple] = None,
                        bins: Union[None, str, int] = 'auto',
                        return_hist_info: bool = False,):
    """
    Returns the hellinger distance score that measures the similarity between two
    distributions. Uses kde to generate pdfs instead of histograms. Produces good
    results without having to fine tune bin size or averaging filter parameters,
    however takes longer to compute.

    Parameters
    ----------------
    ds1:
        array of first dataset
    ds2:
        array of second dataset
    binranges:
        ranges to include in the histograms
    bins:
        the number of bins to use, default is auto
    return_hist_info:
        set to return histograms and edges
    Returns
    -------------
    dist:
        the hellinger distance (n_alts or n_waves x n_feats)
    hist_info:
        additional histogram data that was used to calculate hellinger dist.
        Info includes to real and fake histograms and there shared bin edges.
    """

    if bins is None:
        bins = 'auto'
    if binranges is None:
        hist_args = {'bins': bins, 'density': True}
    else:
        hist_args = {'bins': bins, 'range': binranges, 'density': True}

    if len(ds1) > len(ds2):
        hist, edg = np.histogram(ds1, **hist_args)
    else:
        hist, edg = np.histogram(ds2, **hist_args)

    kernel_1 = stats.gaussian_kde(ds1)
    kernel_2 = stats.gaussian_kde(ds2)

    x_pos = edg[1:] - np.diff(edg) / 2
    h1 = kernel_1(x_pos)
    h2 = kernel_2(x_pos)
    a1 = h1 * np.diff(edg)
    a2 = h2 * np.diff(edg)
    dist = (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(a1) - np.sqrt(a2)) ** 2))

    if return_hist_info:
        return dist, (h1, h2, x_pos)
    else:
        return dist


def hellinger_dist(real, fake, mask=None, bins=None, filter_length=None, z_ranges=None):
    """
    Returns the hellinger distance score between two distributions

    Parameters
    ----------------
    real:
        tensor of real values for a particular alt and bv feat
    fake:
        tensor of generated values for a particular alt and bv feat
    mask:
        mask for bv features
    bins: int
        number of bins to use in histogram calculations
        (If None # of bins will be calculated based on number of samples)
    filter_length: int
        averaging filter length to smooth out noise in histograms
        (If None filter length will be calculated based on number of samples)
        bins: int
        number of bins to use in histogram calculations
        (If None # of bins will be calculated based on number of samples)
    z_ranges: tuple
        upper and lower bounds to evaluate histograms
    Returns
    -------------
    penalty:
        the hellinger distance
    """

    if bins is None:
        bins = max(15, int((len(real)**const.bin_exp)))
    if filter_length is None:
        filter_length = max(2, int(len(real)**const.filter_exp))

    if mask is None:
        r = real.flatten().detach().cpu().numpy()
        f = fake.flatten().detach().cpu().numpy()
    else:
        r = real[mask].flatten().detach().cpu().numpy()
        f = fake[mask].flatten().detach().cpu().numpy()

    if z_ranges is None:
        binranges = (min(np.nanmin(r), np.nanmin(f)),
                     max(np.nanmax(r), np.nanmax(f)))
    else:
        binranges = (max(z_ranges[0], min(np.nanmin(r), np.nanmin(f))),
                     min(z_ranges[1], max(np.nanmax(r), np.nanmax(f))))
        if (f < binranges[0]).all() or (f > binranges[1]).all():
            binranges = (min(np.nanmin(r), np.nanmin(f)), max(np.nanmax(r), np.nanmax(f)))

    args = {'bins': bins, 'range': binranges, 'density': True}
    r_hist, r_edges = np.histogram(r, **args)
    f_hist, f_edges = np.histogram(f, **args)
    if filter_length:
        r_hist = np.convolve(r_hist, np.ones(filter_length), mode='same') / filter_length
        f_hist = np.convolve(f_hist, np.ones(filter_length), mode='same') / filter_length
    dist = (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(r_hist * np.diff(r_edges)) -
                                              np.sqrt(f_hist * np.diff(f_edges))) ** 2))
    return dist


def hellinger_scores_bv(real: np.ndarray,
                        fake: np.ndarray,
                        mask: Union[None, np.ndarray] = None,
                        bins: Union[None, int] = None,
                        filter_length: Union[None, int] = None,
                        return_hist_info: bool = False,
                        z_scale: bool = True,
                        z_scale_input: bool = False,
                        bv_type: str = 'radar'):
    """
    Returns the hellinger distance score that measures how similarity between
    real and generated background variable profiles.

    Parameters
    ----------------
    real: np.ndarray
        tensor of real values for a particular alt and bv feat
    fake: np.ndarray
        tensor of generated values for a particular alt and bv feat
    bins: int
        number of bins to use in histogram calculations
        (If None # of bins will be calculated based on number of samples)
    filter_length: int
        averaging filter length to smooth out noise in histograms
        (If None filter length will be calculated based on number of samples)
    return_hist_info: bool
        set to have function return the histograms and bin edges used which
        were used to calculate the hellinger distance metric.
    z_scale: bool
        used z-scaled values when calculating hellinger distance (recommended)
    z_scale_input: bool
        Set if you are inputting bvs that are
        already z-scaled
    bv_type:
        type of data (radar or lidar)
    Returns
    -------------
    dist:
        the hellinger distance (n_alts x n_feats)
    hist_info
        additional histogram data that was used to calculate hellinger dist.
        Info includes to real and fake histograms and there shared bin edges.
    """
    if mask is None:
        mask = np.ones((real.shape[0], real.shape[1]), dtype=bool)

    if bins is None:
        bins = max(15, int((real.shape[0])**const.bin_exp))
    if filter_length is None:
        filter_length = max(2, int(len(real)**const.filter_exp))

    dists = np.zeros((real.shape[1], real.shape[2]))
    r_hists = np.zeros((bins, real.shape[1], real.shape[2]))
    f_hists = np.zeros((bins, real.shape[1], real.shape[2]))
    edges = np.zeros((bins + 1, real.shape[1], real.shape[2]))

    for i in range(real.shape[1]):
        for j in range(real.shape[2]):
            if z_scale:
                if z_scale_input:
                    r = real[mask[:, i], i, j]
                    f = fake[mask[:, i], i, j]
                else:
                    r = trans.scale_bv(real, bv_type)[0][mask[:, i], i, j]
                    f = trans.scale_bv(fake, bv_type)[0][mask[:, i], i, j]
                binranges = (max(const.bv_z_ranges[j][0], min(np.nanmin(r), np.nanmin(f))),
                             min(const.bv_z_ranges[j][1], max(np.nanmax(r), np.nanmax(f))))
                args = {'bins': bins, 'range': binranges, 'density': True}
            else:
                if z_scale_input:
                    r = trans.get_bv(real, bv_type)[mask[:, i], i, j]
                    f = trans.get_bv(fake, bv_type)[mask[:, i], i, j]
                else:
                    r = real[mask[:, i], i, j]
                    f = fake[mask[:, i], i, j]
                binranges = (max(const.bv_thresholds[j, 0], min(np.nanmin(r), np.nanmin(f))),
                             min(const.bv_thresholds[j, 1], max(np.nanmax(r), np.nanmax(f))))
                args = {'bins': bins, 'range': binranges, 'density': True}
            r_hist, edg = np.histogram(r, **args)
            f_hist, edg = np.histogram(f, **args)
            if filter_length:
                r_hist = np.convolve(r_hist, np.ones(filter_length), mode='same') / filter_length
                f_hist = np.convolve(f_hist, np.ones(filter_length), mode='same') / filter_length
            r_area = r_hist * np.diff(edg)
            f_area = f_hist * np.diff(edg)
            dists[i, j] = (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(r_area) - np.sqrt(f_area)) ** 2))
            r_hists[:, i, j] = r_hist
            f_hists[:, i, j] = f_hist
            edges[:, i, j] = edg

    if return_hist_info:
        return dists, (r_hists, f_hists, edges)
    else:
        return dists


def hellinger_scores_hfp(real: np.ndarray,
                         fake: np.ndarray,
                         r_mask: Union[None, np.ndarray] = None,
                         f_mask: Union[None, np.ndarray] = None,
                         n_bins: Union[None, tuple, int] = None,
                         filter_length: Union[None, int] = None,
                         return_hist_info: bool = False,
                         z_scale: bool = True,
                         z_scale_input: bool = False,):
    """
    Returns the hellinger distance score that measures the similarity between
    real and generated background variable profiles.

    Parameters
    ----------------
    real:
        tensor of real values for a particular alt and bv feat
    fake:
        tensor of generated values for a particular alt and bv feat
    n_bins:
        tensor of real values for a particular alt and bv feat
    filter_length:
        averaging filter length to smooth out histograms
    z_scale: bool
        used z-scaled values (recommended)
    z_scale_input: bool
        Set if you are inputting hfps that are
        already z-scaled
    return_hist_info: bool
        set to have function return the real hist,
        fake hist, and bin edges used in calculation
    Returns
    -------------
    dist:
        the hellinger distance (n_alts or n_waves x n_feats)
    hist_info:
        additional histogram data that was used to calculate hellinger dist.
        Info includes to real and fake histograms and there shared bin edges.
    """
    if r_mask is None:
        r_mask = np.ones((real.shape[0], real.shape[1]), dtype=bool)
    if f_mask is None:
        f_mask = np.ones((fake.shape[0], fake.shape[1]), dtype=bool)

    if n_bins is None:
        n_bins = max(15, int(r_mask.sum() ** const.bin_exp))
    if filter_length is None:
        filter_length = max(2, int(r_mask.sum() ** const.filter_exp))

    dists = np.zeros((real.shape[1], real.shape[2]))
    r_hists = np.zeros((n_bins, real.shape[1], real.shape[2]))
    f_hists = np.zeros((n_bins, real.shape[1], real.shape[2]))
    edges = np.zeros((n_bins + 1, real.shape[1], real.shape[2]))

    for i in range(real.shape[1]):
        for j in range(real.shape[2]):
            if z_scale:
                if z_scale_input:
                    r = real[r_mask[:, i], i, j]
                    f = fake[f_mask[:, i], i, j]
                else:
                    r = trans.scale_hfp(real)[0][r_mask[:, i], i, j]
                    f = trans.scale_hfp(fake)[0][f_mask[:, i], i, j]
                binranges = (max(const.hfp_z_ranges[j][0], min(np.nanmin(r), np.nanmin(f))),
                             min(const.hfp_z_ranges[j][1], max(np.nanmax(r), np.nanmax(f))))
                args = {'bins': n_bins, 'range': binranges, 'density': True}
            else:
                if z_scale_input:
                    r = trans.get_hfp(real)[r_mask[:, i], i, j]
                    f = trans.get_hfp(fake)[f_mask[:, i], i, j]
                else:
                    r = real[r_mask[:, i], i, j]
                    f = fake[f_mask[:, i], i, j]
                binranges = (max(const.hfp_meas_ranges[j][0], min(np.nanmin(r), np.nanmin(f))),
                             min(const.hfp_meas_ranges[j][1], max(np.nanmax(r), np.nanmax(f))))
                args = {'bins': n_bins, 'range': binranges, 'density': True}
            r_hist, edg = np.histogram(r, **args)
            f_hist, edg = np.histogram(f, **args)
            if filter_length:
                r_hist = np.convolve(r_hist, np.ones(filter_length), mode='same') / filter_length
                f_hist = np.convolve(f_hist, np.ones(filter_length), mode='same') / filter_length
            r_area = r_hist * np.diff(edg)
            f_area = f_hist * np.diff(edg)
            dists[i, j] = (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(r_area) -
                                                             np.sqrt(f_area)) ** 2))
            r_hists[:, i, j] = r_hist
            f_hists[:, i, j] = f_hist
            edges[:, i, j] = edg

    if return_hist_info:
        return dists, (r_hists, f_hists, edges)
    else:
        return dists


def hellinger_scores_wtec(real: Union[np.ndarray, torch.Tensor],
                          fake: Union[np.ndarray, torch.Tensor],
                          n_bins: Union[None, int] = None,
                          filter_length: Union[None, int] = None,
                          z_scale: bool = False,
                          z_scale_inputs: bool = False,
                          tid_type: str = 'SSTIDs',
                          use_kde: bool = False,
                          return_hist_info: bool = False,):
    """
    Returns the hellinger distance score that measures the similarity between
    real and generated tec wave.

    Parameters
    ----------------
    real:
        array of real tec waves
    fake:
        array of fake/generated tec waves
    n_bins:
        number of bins to use during hellinger score calculation
    filter_length:
        averaging filter length to smooth out histograms
    z_scale: bool
        used z-scaled values (recommended)
    z_scale_inputs: bool
        Set if you are inputting hfps that are
        already z-scaled
    tid_type: str
        specify dataset type for z-scaling
    use_kde: bool
        use kernel density estimation to generate PDFs
    return_hist_info: bool
        set to have function return the real hist,
        fake hist, and bin edges used in calculation
    Returns
    -------------
    dist:
        the hellinger distance (1 x n_feats)
    hist_info:
        additional histogram data that was used to calculate hellinger dist.
        Info includes to real and fake histograms and there shared bin edges.
    """

    if n_bins is None:
        n_bins = max(15, int(len(real) ** const.bin_exp))
    if filter_length is None:
        filter_length = max(2, int(len(real) ** const.filter_exp))

    dists = np.zeros(real.shape[1])
    if use_kde:
        r_hists, f_hists, edges = [], [], []
    else:
        r_hists = np.zeros((n_bins, real.shape[1]))
        f_hists = np.zeros((n_bins, real.shape[1]))
        edges = np.zeros((n_bins + 1, real.shape[1]))

    for i in range(real.shape[1]):
        if z_scale:
            if z_scale_inputs:
                r = real[:, i]
                f = fake[:, i]
            else:
                r = trans.scale_wtec(real, tid_type=tid_type)[0][:, i]
                f = trans.scale_wtec(fake, tid_type=tid_type)[0][:, i]
            binranges = (max(const.wtec_dict[tid_type]['z_ranges'][i][0], min(np.nanmin(r), np.nanmin(f))),
                         min(const.wtec_dict[tid_type]['z_ranges'][i][1], max(np.nanmax(r), np.nanmax(f))))
            if (f < binranges[0]).all() or (f > binranges[1]).all():
                binranges = (min(np.nanmin(r), np.nanmin(f)), max(np.nanmax(r), np.nanmax(f)))
            args = {'bins': n_bins, 'range': binranges, 'density': True}
        else:
            if z_scale_inputs:
                r = trans.get_wtec(real, tid_type=tid_type)[:, i]
                f = trans.get_wtec(fake, tid_type=tid_type)[:, i]
            else:
                r = real[:, i]
                f = fake[:, i]
            binranges = (max(const.wtec_dict[tid_type]['meas_ranges'][i][0], min(np.nanmin(r), np.nanmin(f))),
                         min(const.wtec_dict[tid_type]['meas_ranges'][i][1], max(np.nanmax(r), np.nanmax(f))))
            if (f < binranges[0]).all() or (f > binranges[1]).all():
                binranges = (min(np.nanmin(r), np.nanmin(f)), max(np.nanmax(r), np.nanmax(f)))
            args = {'bins': n_bins, 'range': binranges, 'density': True}

        if use_kde:
            dists[i], (r_hist, f_hist, edg) = hellinger_score_kde(r, f, binranges=binranges, return_hist_info=True)
            r_hists.append(r_hist)
            f_hists.append(f_hist)
            edges.append(edg)
        else:
            r_hist, edg = np.histogram(r, **args)
            f_hist, edg = np.histogram(f, **args)
            if filter_length:
                r_hist = np.convolve(r_hist, np.ones(filter_length), mode='same') / filter_length
                f_hist = np.convolve(f_hist, np.ones(filter_length), mode='same') / filter_length
            r_area = r_hist * np.diff(edg)
            f_area = f_hist * np.diff(edg)
            dists[i] = (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(r_area) -
                                                          np.sqrt(f_area)) ** 2))
            r_hists[:, i] = r_hist
            f_hists[:, i] = f_hist
            edges[:, i] = edg

    if return_hist_info:
        return dists, (r_hists, f_hists, edges)
    else:
        return dists
