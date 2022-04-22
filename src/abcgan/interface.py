"""
Code for top level interface.

This code is added to the main package level in __init__.py
"""
import numpy as np
import abcgan.constants as const
import abcgan.transforms as trans
from abcgan import persist
from abcgan.mask import mask_altitude
from abcgan.transforms import compute_valid, compute_valid_hfp
import torch
import h5py
from tqdm import tqdm
from warnings import warn


def generate(drivers, measurements=None,
             driver_names=const.driver_names, n_alt=const.max_alt,
             bv_model='bv_gan', hfp_model='hfp_gan', bv_type='radar',
             generate_hfps=False, return_z_scale=False, cuda_index=None,
             verbose=1):
    """
    Generate background variable profiles and HFP waves
    consistent with the historical distribution.

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    driver_names: list
        list of names of driving parameters
    measurements: np.ndarray, optional
        n_samples x n_alt_in x n_meas input list of altitude measurements,
        n_alt_in should be less than n_alt. These represent fixed
        measurements for the lowest altitudes to condition on.
        Usually left as default (None)
    n_alt: int, optional
        number of altitude measurements to draw, defaults to max_alt
    return_z_scale: bool, optional
        set to have the function return z scaled feature data
    bv_model: str, optional
        name of bv GAN to use
    bv_type: str. optional
        name of the type of background variables to use (lidar or radar)
    hfp_model: str, optional
        name of hfp GAN to use
    generate_hfps: bool, optional
        Set to generate hfps and return generated hfps
    cuda_index: int, optional
        GPU index to use when generating BVs and HFPs
    verbose: bool, optional
        set to show loading bar
    Returns
    -------------
    samples: (np.ndarray, np.ndarray, np.ndarray)
        1) n_samples x n_alt x n_bvs output measurements at each requested altitude.
        2) n_samples x n_hfps generated hfp waves
        3) n_sample probabilities that the generated wave is present
    """

    with torch.no_grad():
        disable1 = bool(verbose < 1)
        if n_alt > const.max_alt and bv_type == 'radar':
            raise ValueError(f"Requested {n_alt} altitudes but only {const.max_alt}"
                             f" can be simulated for {bv_type}.")
        elif n_alt > const.max_alt_lidar and bv_type == 'lidar':
            raise ValueError(f"Requested {n_alt} altitudes but only {const.max_alt_lidar}"
                             f" can be simulated for {bv_type}.")
        if bv_type == 'lidar':
            n_bv = const.n_lidar_bv
        else:
            n_bv = const.n_bv

        if measurements is None:
            # put placeholder measurements if none provided
            measurements = np.zeros((drivers.shape[0], 0, n_bv))

        n_batch = drivers.shape[0]
        n_alt_start = measurements.shape[1]

        # verify the correct shapes for the inputs
        if drivers.shape != (n_batch, len(driver_names)):
            raise ValueError(f"driver and driver_names must have the "
                             f"same length ({drivers.shape[-1]} != {len(driver_names)}")
        if measurements.shape != (n_batch, n_alt_start, n_bv):
            raise ValueError(f"Measurement shape must be n_batch x "
                             f"{n_alt_start} x {n_bv}.")

        # z scale inputs and place into tensors
        driver_feat = trans.scale_driver(drivers, driver_names=driver_names)
        bv_feat, valid_mask = trans.scale_bv(measurements, bv_type)
        driver_feat = torch.tensor(driver_feat, dtype=torch.float)
        bv_feat = torch.tensor(bv_feat, dtype=torch.float)
        bv_feat, alt_mask = mask_altitude(bv_feat)
        hfp_feats = torch.zeros(len(drivers), const.n_waves, const.n_hfp_feat)
        G_b = torch.zeros(len(drivers))

        if torch.cuda.is_available() and cuda_index is not None:
            device = torch.device('cuda:' + str(cuda_index))
            if len(drivers) >= 500:
                batch_size = 500
            else:
                batch_size = len(drivers)
        else:
            device = torch.device('cpu')
            if len(drivers) >= 100:
                batch_size = 100
            else:
                batch_size = len(drivers)

        batch_idxs = np.hstack((np.arange(0, len(drivers), step=batch_size), [len(drivers)]))

        # Load bv models
        bv_gen, _ = persist.recreate(name=bv_model)
        if bv_gen.transformer.dr_emb.shape[0] != driver_feat.shape[-1]:
            raise ValueError(f"Model must be trained with "
                             f"{driver_feat.shape[-1]} drivers.")
        bv_gen.to(device)
        bv_gen.eval()

        # iteratively build altitude profile
        for i in tqdm(range(len(batch_idxs) - 1), desc='Generate BV Profile', disable=disable1):
            for i_alt in range(n_alt_start, n_alt):
                dr_src = driver_feat[batch_idxs[i]:batch_idxs[i + 1]].to(device)
                bv_src = bv_feat[batch_idxs[i]:batch_idxs[i + 1]].to(device)

                bv_out = bv_gen(dr_src, bv_src)
                # fill in next altitude
                bv_feat[batch_idxs[i]:batch_idxs[i + 1], i_alt, :] = bv_out[:, i_alt, :].detach().cpu()
        G_bv_feats = bv_feat[:, :n_alt, :]
        G_bvs = trans.get_bv(bv_feat[:, :n_alt, :].numpy(), bv_type)
        if measurements is not None:
            G_bvs[:, :n_alt_start, :] = measurements
        bv_gen.cpu()

        if generate_hfps:
            # Load hfp GAN
            hfp_gen, _ = persist.recreate(name=hfp_model)
            hfp_gen.to(device)
            hfp_gen.eval()

            # Generate waves
            for i in tqdm(range(len(batch_idxs) - 1), desc='Generate HFP Batches', disable=disable1):
                dr_src = driver_feat[batch_idxs[i]:batch_idxs[i + 1]].to(device)
                gbv_src = G_bv_feats[batch_idxs[i]:batch_idxs[i + 1]].to(device)
                hfp_tgt = hfp_feats[batch_idxs[i]:batch_idxs[i + 1]].to(device)

                hfp_out, gb = hfp_gen(dr_src, gbv_src, hfp_tgt)
                hfp_feats[batch_idxs[i]:batch_idxs[i + 1], ...] = hfp_out.detach().cpu()
                G_b[batch_idxs[i]:batch_idxs[i + 1]] = gb.detach().cpu()
            G_hfps_feats = hfp_feats.numpy()
            G_hfps = trans.get_hfp(hfp_feats.numpy())
            G_b = G_b.numpy()

        if generate_hfps:
            if return_z_scale:
                return G_bv_feats.numpy(), G_hfps_feats, G_b
            else:
                return G_bvs, G_hfps, G_b
        else:
            if return_z_scale:
                return G_bv_feats.numpy()
            else:
                return G_bvs


def generate_multi(drivers, bvs=None, n_repeat=10,
                   bv_model='bv_gan', hfp_model='hfp_gan',
                   bv_type='radar', generate_hfps=False,
                   cuda_index=None, verbose=1):
    """
    Generate multiple background variable profiles and HFP waves
    consistent with the historical distribution for each driver sample

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    bvs: np.ndarray, optional
        n_samples x n_alt_in x n_meas input list of altitude measurements,
        n_alt_in should be less than n_alt. These represent fixed
        measurements for the lowest altitudes to condition on.
        Usually left as default (None)
    n_repeat: int, optional
        number of bv profiles/waves to generate for each driver sample
    bv_model: str, optional
        name of bv GAN to use
    bv_type: str. optional
        name of the type of background variables to use (lidar or radar)
    hfp_model: str, optional
        name of hfp GAN to use
    generate_hfps: bool, optional
        Set to generate hfps and return generated hfps
    cuda_index: int, optional
        GPU index to use when generating BVs and HFPs
    verbose: bool, optional
        set to show loading bar
    Returns
    -------------
    samples: (np.ndarray, np.ndarray, np.ndarray)
        1) (n_samples x n_repeat x n_alt x n_bvs) output measurements at each requested altitude.
        2) (n_samples x n_repeat x 1 x n_hfps) generated hfp waves
        3) (n_sample x n_repeat) probabilities that the generated wave is present
    """
    disable = bool(verbose < 1)
    G_bvs = np.zeros((len(drivers), n_repeat, const.max_alt, const.n_bv))
    G_hfps = np.zeros((len(drivers), n_repeat, const.n_waves, const.n_hfp))
    G_b = np.zeros((len(drivers), n_repeat))

    for i in tqdm(range(len(drivers)), desc='Generating Samples', disable=disable):
        sampled_driver = drivers[[i], ...].repeat(n_repeat, 0)
        if bvs is not None:
            sampled_bv = bvs[[i], ...].repeat(n_repeat, 0)
        else:
            sampled_bv = None
        if generate_hfps:
            gen_data = generate(sampled_driver,
                                measurements=sampled_bv,
                                bv_model=bv_model,
                                hfp_model=hfp_model,
                                bv_type=bv_type,
                                generate_hfps=True,
                                cuda_index=cuda_index,
                                verbose=False)
            G_bvs[i, ...] = gen_data[0]
            G_hfps[i, ...] = gen_data[1]
            G_b[i, ...] = gen_data[2]
        else:
            G_bvs[i, ...] = generate(sampled_driver,
                                     measurements=sampled_bv,
                                     bv_model=bv_model,
                                     hfp_model=hfp_model,
                                     bv_type=bv_type,
                                     generate_hfps=False,
                                     cuda_index=cuda_index,
                                     verbose=False)
    if generate_hfps:
        return G_bvs, G_hfps, G_b
    else:
        return G_bvs


def discriminate(drivers, bvs, hfps=None,
                 driver_names=const.driver_names,
                 bv_model='bv_gan', hfp_model='hfp_gan',
                 bv_type='radar'):
    """
    Score how well the measurements match with historical observations.

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    driver_names: list
        list of names of driving parameters
    bvs: np.ndarray
        n_samples x n_alt_in x n_meas input list of altitude measurements,
        n_alt_in should be less than max_alt.
    hfps: np.ndarray, optional
        n_samples x n_wave x n_hfps input list of wave measurements,
    bv_model: str, optional
        name of bv model to use
    hfp_model: str, optional
        name of model hfp to use
    bv_type: str. optional
        name of the type of background variables to use (lidar or radar)
    Returns
    -------------
    scores: (np.ndarray, np.ndarray)
        1) n_samples x n_alt bv normalcy scores in the range [0, 1.0].
        2) n_samples hfp wave normalcy scores in the range [0, 1.0].
    """
    warn(f'The discriminate function is deprecated. Does not produce credible results '
         f'outside of the training process. Please use the anomaly modules in place of'
         f'the discriminator', DeprecationWarning, stacklevel=2)
    with torch.no_grad():
        n_batch, n_alt = bvs.shape[:2]
        _, bv_crit = persist.recreate(name=bv_model)
        bv_crit.eval()

        driver_feat = trans.scale_driver(drivers, driver_names)
        bv_feat, _ = trans.scale_bv(bvs, bv_type)

        if bv_crit.transformer.dr_emb.shape[0] != driver_feat.shape[-1]:
            raise ValueError(f"Model must be trained with "
                             f"{driver_feat.shape[-1]} drivers.")

        driver_feat = torch.tensor(driver_feat, dtype=torch.float)
        bv_feat = torch.tensor(bv_feat, dtype=torch.float)
        bv_feat, alt_mask = mask_altitude(bv_feat)

        # Get bvs scores
        bv_scores = bv_crit(bv_feat, driver_feat, bv_feat, ~alt_mask)
        bv_scores = bv_scores.view(n_batch, -1)[:, :n_alt].cpu().numpy()

        if hfps is not None:
            if hfps.shape[-1] != const.n_hfp:
                raise ValueError(f"HFPs data must have "
                                 f"{const.n_hfp} features.")

            _, hfp_crit = persist.recreate(name=hfp_model)
            hfp_crit.eval()
            hfp_feat, _ = trans.scale_hfp(hfps)
            hfp_feat = torch.tensor(hfp_feat, dtype=torch.float)

            # Get hfp scores
            hfp_scores = hfp_crit(driver_feat, bv_feat, hfp_feat, hfp_feat, ~alt_mask)
            hfp_scores = hfp_scores.view(n_batch, -1).cpu().numpy()

        if hfps is None:
            return bv_scores
        else:
            return bv_scores, hfp_scores


def estimate_drivers(drivers, model='dr_gan'):
    """
    Predict drivers 2 hours into the future driver GAN model. Used for real-time
    background predictions using drivers from 2 hours ago.

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    model: str, optional
        name of model to use

    Returns
    -------------
    predicted_drivers: np.ndarray
        estimation of driver features two hours from the drivers inputted
    """
    driver_feats = torch.tensor(trans.scale_driver(drivers), dtype=torch.float)
    dr_gen, _ = persist.recreate(name=model)
    dr_gen.eval()

    with torch.no_grad():
        predicted_feats = dr_gen(driver_feats)

    predicted_drivers = trans.get_driver(predicted_feats.cpu().numpy())
    return predicted_drivers


def hellinger_scores_bv(real, fake, mask=None, bins=None, filter_length=None,
                        return_hist_info=False, z_scale=True, bv_type='radar'):
    """
    Returns the hellinger distance score that measures how similarity between
    real and generated background variable profiles.

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
    bv_type:
        type of data (radar or lidar)

    Returns
    -------------
    dist:
        the hellinger distance (n_alts x n_feats)
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
                r = trans.scale_bv(real, bv_type)[0][mask[:, i], i, j]
                f = trans.scale_bv(fake, bv_type)[0][mask[:, i], i, j]
                args = {'bins': bins, 'range': (-3, 3), 'density': True}
            else:
                r = real[mask[:, i], i, j]
                f = fake[mask[:, i], i, j]
                args = {'bins': bins, 'range': (const.bv_thresholds[j, 0], const.bv_thresholds[j, 1]),
                        'density': True}
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


def hellinger_scores_hfp(real, fake, r_mask=None, f_mask=None,
                         bins=None, filter_length=None, z_scale=True,
                         return_hist_info=False):
    """
    Returns the hellinger distance score that measures the similarity between
    real and generated background variable profiles.

    real:
        tensor of real values for a particular alt and bv feat
    fake:
        tensor of generated values for a particular alt and bv feat
    bins:
        tensor of real values for a particular alt and bv feat
    filter_length:
        averaging filter length to smooth out histograms
    z_scale: bool
        used z-scaled values (recommended)
    return_hist_info: bool
        set to have function return the real hist,
        fake hist, and bin edges used in calculation

    Returns
    -------------
    dist:
        the hellinger distance (n_alts or n_waves x n_feats)
    """
    if r_mask is None:
        r_mask = np.ones((real.shape[0], real.shape[1]), dtype=bool)
    if f_mask is None:
        f_mask = np.ones((fake.shape[0], fake.shape[1]), dtype=bool)

    if bins is None:
        bins = max(15, int(r_mask.sum() ** const.bin_exp))
    if filter_length is None:
        filter_length = max(2, int(r_mask.sum() ** const.filter_exp))

    dists = np.zeros((real.shape[1], real.shape[2]))
    r_hists = np.zeros((bins, real.shape[1], real.shape[2]))
    f_hists = np.zeros((bins, real.shape[1], real.shape[2]))
    edges = np.zeros((bins + 1, real.shape[1], real.shape[2]))

    for i in range(real.shape[1]):
        for j in range(real.shape[2]):
            if z_scale:
                r = trans.scale_hfp(real)[0][r_mask[:, i], i, j]
                f = trans.scale_hfp(fake)[0][f_mask[:, i], i, j]
                args = {'bins': bins, 'range': const.hfp_z_ranges[j], 'density': True}
            else:
                r = real[r_mask[:, i], i, j]
                f = fake[f_mask[:, i], i, j]
                args = {'bins': bins,
                        # 'range': (min(np.min(r), np.min(f)), max(np.max(r), np.max(f))),
                        'range': (const.hfp_thresholds[j, 0], const.hfp_thresholds[j, 1]),
                        'density': True}
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


def stack_drivers(driver_dict, driver_names=const.driver_names):
    """
    Stacks drivers in appropriate format.

    This function is provided for convenience.


    Parameters
    ----------------
    driver_dict: dict
        Dictionary mapping names of drivers to the numpy arrays
        with values for those drivers. Each array has a single
        dimension of the same length n_samples.
        Can also use an `h5py.Group`.
    driver_names: list
        names of the drivers to load

    Valid names for drivers can be found at `abcgan.driver_names`

    Raises
    ------------------
    ValueError:
        If the driver values have the wrong type or shape.
    KeyError:
        If one of the required drivers is missing.
    """
    if isinstance(driver_dict, h5py.Group):
        driver_dict = {k: v[()] for k, v in driver_dict.items()}
    shp = None
    for v in driver_dict.values():
        if not isinstance(v, np.ndarray):
            raise ValueError(f"Values in driver_dict must have"
                             f" type np.ndarray not {type(v)}.")
        if shp is None:
            shp = v.shape
            if len(shp) != 1:
                raise ValueError("Driver dict values must have only one"
                                 " dimension for the number of samples.")
        if shp != v.shape:
            raise ValueError("All values in driver_dict must have"
                             " the same length.")
    return np.stack([driver_dict[k] for k in driver_names],
                    axis=-1)


def stack_bvs(bv_dict, bv_type='radar'):
    """
    Stacks drivers in appropriate format.

    This function is provided for convenience.

    Parameters
    ----------------
    bv_dict: dict
        Dictionary mapping names of background variables
        to numpy arrays with values for those bvs. Each
        array should have shape n_sapmles x n_altitudes.
        Can also use `h5py.Group`.
    bv_type: str
        string specifying weather to stack radar or
        lidar data

    Valid names for drivers can be found at `abcgan.bv_names`

    Raises
    ------------------
    ValueError:
        If the input shape of the bv dict values is not corrects
    KeyError:
        If one of the required bvs is missing.
    """
    if isinstance(bv_dict, h5py.Group):
        bv_dict = {k: v[()] for k, v in bv_dict.items()}
    shp = None
    for v in bv_dict.values():
        if not isinstance(v, np.ndarray):
            raise ValueError(f"Values in bv_dict must have"
                             f" type np.ndarray not {type(v)}.")
        if shp is None:
            shp = v.shape
            if len(shp) != 2:
                raise ValueError("BV dict values must have 2 "
                                 "dimensions [n_samples x n_altitudes].")
        if shp != v.shape:
            raise ValueError("All values in bv_dict must have the"
                             " same shape.")
    if bv_type == 'lidar':
        bvs = np.stack([bv_dict[k] for k in const.lidar_bv_names],
                       axis=-1)[:, :const.max_alt_lidar, :]
    else:
        bvs = np.stack([bv_dict[k] for k in const.bv_names],
                       axis=-1)[:, 26:26 + const.max_alt, :]
    return bvs


def load_h5_data(fname, bv_type='radar', load_hfp=False, n_samples=None):
    """
    loads and returns external drivers, background variables, HFP waves and data
    mask all aligned in time with outlier/invalid data filtered out

    Parameters
    -------------
    fname: str
        name of h5 file to load the data from
    bv_type: str. optional
        name of the type of background variables to use (lidar or radar)
    load_hfp: bool. optional
        set to load HFP waves along with the background variables
    n_samples: int. optional
        number of samples to load (None to load all samples)
    Returns
    -------------
    drivers: np.ndarray
        (n_samples x n_dr) external drivers.
    bvs: np.ndarray
        (n_samples x n_alt x n_bv) background variables.
    alt_mask: np.ndarray
        (n_samples x n_alt) background variables alt mask.
    hfps: np.ndarray
        (n_samples x 1 x n_hpf) HPF waves.
    wave_mask: np.ndarray
        (n_samples x 1) HFP wave mask.
    unix_time: np.ndarray
        (n_samples, ) time stamp of each sample
    """
    with h5py.File(fname, 'r') as f:
        # --------------------------------------
        # Read driver and time stamps from file
        # --------------------------------------
        dr_dict = f['Drivers']
        unix_time = f['UnixTime'][()]
        drivers = np.stack([dr_dict[driver_name][:]
                            for driver_name in const.driver_names
                            if driver_name in dr_dict.keys()],
                           axis=-1)
        # --------------------------------------
        # Read background variable data from file
        # --------------------------------------
        bv_dict = f['BackgroundValues']
        if bv_type == 'lidar':
            bvs = np.stack([bv_dict[bv_name][:]
                            for bv_name in const.lidar_bv_names],
                           axis=-1)[:, :const.max_alt_lidar, :]
            bv_thresholds = const.lidar_thresholds
        else:
            bvs = np.stack([bv_dict[bv_name][:]
                            for bv_name in const.bv_names],
                           axis=-1)[:, 26:26 + const.max_alt]
            bv_thresholds = const.bv_thresholds

        if load_hfp:
            # --------------------------------------
            # Read HFP data from file
            # --------------------------------------
            hfp_dict = f['HFPValues']
            hfps = np.stack([hfp_dict[hfp_name]
                             for hfp_name in const.hfp_names
                             if hfp_name in hfp_dict.keys()],
                            axis=-1)
            hfps = hfps[:, None, :]

    # Get valid bvs and altitude mask
    valid_bv_mask = compute_valid(bvs, bv_thresholds)

    if load_hfp:
        valid_hfp_mask = compute_valid_hfp(hfps)
        valid_mask = valid_bv_mask & valid_hfp_mask & ~(np.isnan(drivers).any(-1))
    else:
        valid_mask = valid_bv_mask & ~(np.isnan(drivers).any(-1))

    # Filter out any invalid samples
    if n_samples is not None:
        drivers = drivers[valid_mask][:n_samples]
        bvs = bvs[valid_mask][:n_samples]
        unix_time = unix_time[valid_mask][:n_samples]
    else:
        drivers = drivers[valid_mask]
        bvs = bvs[valid_mask]
        unix_time = unix_time[valid_mask]

    # Get altitude mask for bvs
    _, alt_mask = mask_altitude(torch.tensor(bvs, dtype=torch.float))
    alt_mask = alt_mask.detach().numpy()

    if load_hfp:
        # Filter out invalid samples and get wave mask
        if n_samples is not None:
            hfps = hfps[valid_mask][:n_samples]
        else:
            hfps = hfps[valid_mask]
        wave_mask = ~(np.isnan(hfps).any(-1))

        return drivers, bvs, alt_mask, hfps, wave_mask, unix_time

    else:
        return drivers, bvs, alt_mask, unix_time
