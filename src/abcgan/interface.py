"""
Code for top level interface.

This code is added to the main package level in __init__.py
"""
import numpy as np
import abcgan.constants as const
import abcgan.transforms as trans
from abcgan import persist
from abcgan.mask import mask_altitude
from abcgan.transforms import compute_valid, compute_valid_hfp, compute_valid_wtec
import torch
import h5py
from tqdm import tqdm
from warnings import warn
from typing import List, Union, Tuple
from torch.utils.data import TensorDataset, DataLoader


def generate_wtec(drivers,
                  driver_names: list = const.driver_names,
                  mean_replace_drs: Union[None, List[str]] = None,
                  wtec_model: str = const.wtec_default_model,
                  dataset_name: Union[None, str] = None,
                  model_dir: Union[None, str] = None,
                  return_z_scale: bool = False,
                  cuda_index: Union[None, int] = None,
                  verbose: int = 1):
    """
    Generate background variable profiles and HFP waves
    consistent with the historical distribution.

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    driver_names: list
        list of names of driving parameters
    mean_replace_drs: list
        list of driver that will set to its average, i.e. z-scaled value of zero
    return_z_scale: bool, optional
        set to have the function return z scaled feature data
    wtec_model: str, optional
        name of WTEC GAN to use
    dataset_name: str
        specify dataset type for z-scaling
    model_dir: str, optional
        directory to load model from
    cuda_index: int, optional
        GPU index to use when generating BVs and HFPs
    verbose: bool, optional
        set to show loading bar
    Returns
    -------------
    samples: np.ndarray
        1) n_samples x n_wtec output measurements
    """

    with torch.no_grad():
        disable1 = bool(verbose < 1)
        n_batch = drivers.shape[0]
        # verify the correct shapes for the inputs
        if drivers.shape != (n_batch, len(driver_names)):
            raise ValueError(f"driver and driver_names must have the "
                             f"same length ({drivers.shape[-1]} != {len(driver_names)}")

        # z scale inputs and place into tensors
        driver_feat = trans.scale_driver(drivers, driver_names=driver_names)
        driver_feat = torch.tensor(driver_feat, dtype=torch.float)

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
        if mean_replace_drs is not None and len(mean_replace_drs) > 0:
            replace_dr_idxs = np.hstack([const.dr_feat_map[n] for n in mean_replace_drs])
        else:
            replace_dr_idxs = np.array([])

        # Load bv models
        if model_dir is None:
            wtec_gen, _ = persist.recreate(name=wtec_model)
        else:
            wtec_gen, _ = persist.recreate(name=wtec_model, dir_path=model_dir)

        if wtec_gen.cond_dim != driver_feat.shape[-1]:
            raise ValueError(f"Model must be trained with "
                             f"{driver_feat.shape[-1]} drivers.")
        wtec_gen.to(device)
        wtec_feats = torch.zeros((len(driver_feat), const.n_wtec_feat), device=device)
        wtec_gen.eval()

        # iteratively build altitude profile
        for i in tqdm(range(len(batch_idxs) - 1), desc='Generating TEC Waves', disable=disable1):
            dr_src = driver_feat[batch_idxs[i]:batch_idxs[i + 1]].to(device)
            dr_src[..., replace_dr_idxs] = 0.0
            wtec_feats[batch_idxs[i]:batch_idxs[i + 1], :] = wtec_gen(dr_src)

        G_wtec_feats = wtec_feats.detach().cpu().numpy()
        G_wtecs = trans.get_wtec(wtec_feats.detach().cpu().numpy(), dataset_name=dataset_name)
        wtec_gen.cpu()

        if return_z_scale:
            return G_wtec_feats
        else:
            return G_wtecs


def generate_multi_wtec(drivers: np.ndarray,
                        n_repeat: int = 10,
                        wtec_model: str = const.wtec_default_model,
                        model_dir: Union[None, str] = None,
                        dataset_name=const.wtec_default_dataset,
                        cuda_index: Union[None, int] = None,
                        verbose: int = 0):
    """
    Generate multiple background variable profiles and HFP waves
    consistent with the historical distribution for each driver sample

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    n_repeat: int, optional
        number of waves to generate for each driver sample
    wtec_model: str, optional
        name of WTEC GAN to use
    model_dir: str, optional
        directory to load model from
    dataset_name: str
        specify dataset type for z-scaling
    cuda_index: int, optional
        GPU index
    verbose: bool, optional
        set to show loading bar
    Returns
    -------------
    samples: np.ndarray
        (n_samples x n_repeat x n_wtec) output wtec measurements.
    """
    disable = bool(verbose < 1)
    G_wtec = np.zeros((len(drivers), n_repeat, const.n_wtec))

    for i in tqdm(range(len(drivers)), desc='Generating Samples', disable=disable):
        sampled_driver = drivers[[i], ...].repeat(n_repeat, 0)
        G_wtec[i, ...] = generate_wtec(sampled_driver,
                                       driver_names=const.wtec_dr_names,
                                       dataset_name=dataset_name,
                                       wtec_model=wtec_model,
                                       model_dir=model_dir,
                                       cuda_index=cuda_index,
                                       verbose=False)
    return G_wtec


def generate_bvs(drivers: np.ndarray,
                 bv_measurements: Union[None, np.ndarray] = None,
                 driver_names: List[str] = const.driver_names,
                 mean_replace_drs: Union[None, List[str]] = None,
                 n_alt: int = const.max_alt,
                 bv_model: str = 'bv_gan',
                 model_dir: Union[None, str] = None,
                 bv_type: str = 'radar',
                 return_z_scale: bool = False,
                 cuda_index: Union[None, int] = None,
                 verbose: int = 1):
    """
    Generate background variable profiles
    consistent with the historical distribution.

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    driver_names: list
        list of names of driving parameters
    mean_replace_drs: list
        list of driver that will set to its average, i.e. z-scaled value of zero
    bv_measurements: np.ndarray, optional
        n_samples x n_alt_in x n_meas input list of altitude measurements,
        n_alt_in should be less than n_alt. These represent fixed
        measurements for the lowest altitudes to condition on.
    n_alt: int, optional
        number of altitude measurements to draw, defaults to max_alt
    return_z_scale: bool, optional
        set to have the function return z scaled feature data
    bv_model: str, optional
        name of bv GAN to use
    model_dir: str, optional
        directory to load model from
    bv_type: str. optional
        name of the type of background variables to use (lidar or radar)
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

        if bv_measurements is None:
            # put placeholder measurements if none provided
            bv_measurements = np.zeros((drivers.shape[0], 0, n_bv))

        n_batch = drivers.shape[0]
        n_alt_start = bv_measurements.shape[1]

        # verify the correct shapes for the inputs
        if drivers.shape != (n_batch, len(driver_names)):
            raise ValueError(f"driver and driver_names must have the "
                             f"same length ({drivers.shape[-1]} != {len(driver_names)}")
        if bv_measurements.shape != (n_batch, n_alt_start, n_bv):
            raise ValueError(f"Measurement shape must be n_batch x "
                             f"{n_alt_start} x {n_bv}.")

        # z scale inputs and place into tensors
        driver_feat = trans.scale_driver(drivers, driver_names=driver_names)
        driver_feat = torch.tensor(driver_feat, dtype=torch.float)
        G_bv_feats, valid_mask = trans.scale_bv(bv_measurements, bv_type)
        G_bv_feats = torch.tensor(G_bv_feats, dtype=torch.float)
        G_bv_feats, alt_mask = mask_altitude(G_bv_feats)

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
        if mean_replace_drs is not None and len(mean_replace_drs) > 0:
            missing_dr_idxs = np.hstack([const.dr_feat_map[n] for n in mean_replace_drs])
        else:
            missing_dr_idxs = np.array([])

        if n_alt_start < n_alt:
            # Load bv models
            if model_dir is None:
                bv_gen, _ = persist.recreate(name=bv_model)
            else:
                bv_gen, _ = persist.recreate(name=bv_model, dir_path=model_dir)
            if bv_gen.transformer.dr_emb.shape[0] != driver_feat.shape[-1]:
                raise ValueError(f"Model must be trained with "
                                 f"{driver_feat.shape[-1]} drivers.")
            bv_gen.to(device)
            bv_gen.eval()

            # iteratively build altitude profile
            for i in tqdm(range(len(batch_idxs) - 1), desc='Generate BV Profile', disable=disable1):
                for i_alt in range(n_alt_start, n_alt):
                    dr_src = driver_feat[batch_idxs[i]:batch_idxs[i + 1]].to(device)
                    bv_src = G_bv_feats[batch_idxs[i]:batch_idxs[i + 1]].to(device)

                    dr_src[:, missing_dr_idxs] = 0.0

                    bv_out = bv_gen(dr_src, bv_src)
                    # fill in next altitude
                    G_bv_feats[batch_idxs[i]:batch_idxs[i + 1], i_alt, :] = bv_out[:, i_alt, :].detach().cpu()
            G_bvs = trans.get_bv(G_bv_feats[:, :n_alt, :].numpy(), bv_type)
            G_bvs[:, :n_alt_start, :] = bv_measurements[:, :n_alt_start]
            bv_gen.cpu()
        else:
            print("No Altitude profiles to generate based on inputted bv conditioning")
            G_bvs = bv_measurements.copy()

        if return_z_scale:
            return G_bv_feats.numpy()
        else:
            return G_bvs


def generate_hfps(drivers: np.ndarray,
                  bv_measurements: np.ndarray,
                  driver_names: List[str] = const.driver_names,
                  mean_replace_drs: Union[None, List[str]] = None,
                  hfp_model: str = 'hfp_gan',
                  model_dir: Union[None, str] = None,
                  return_z_scale: bool = False,
                  cuda_index: Union[None, int] = None,
                  verbose: int = 1):
    """
    Generate background variable profiles and HFP waves
    consistent with the historical distribution.

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    driver_names: list
        list of names of driving parameters
    mean_replace_drs: list
        list of driver that will set to its average, i.e. z-scaled value of zero
    bv_measurements: np.ndarray, optional
        n_samples x n_alt_in x n_meas input list of altitude measurements,
        n_alt_in should be less than n_alt. These represent fixed
        measurements for the lowest altitudes to condition on.
    return_z_scale: bool, optional
        set to have the function return z scaled feature data
    hfp_model: str, optional
        name of hfp GAN to use
    model_dir: str, optional
        directory to load model from
    cuda_index: int, optional
        GPU index to use when generating BVs and HFPs
    verbose: bool, optional
        set to show loading bar
    Returns
    -------------
    samples: (np.ndarray, np.ndarray)
        1) n_samples x n_hfps generated hfp waves
        2) n_sample probabilities that the generated wave is present
    """

    with torch.no_grad():
        disable1 = bool(verbose < 1)
        if bv_measurements.shape[-1] != const.n_bv:
            raise ValueError(f"HFP GAN does not support Lidar BV measurements. Must input "
                             f"{const.n_bv} radar bv features")
        if bv_measurements.shape[1] > const.max_alt or bv_measurements.shape[1] <= 0:
            raise ValueError(f"Background measruements must have least one alt bin, but "
                             f"no more then {const.max_alt}.")
        if len(drivers) != len(bv_measurements):
            raise ValueError(f"driver and bv measurements"
                             f"must have the same number of sample "
                             f"({drivers.shape[0]} != {len(bv_measurements)})")
        n_bv = const.n_bv
        n_batch = drivers.shape[0]
        n_alt_start = bv_measurements.shape[1]

        # verify the correct shapes for the inputs
        if drivers.shape != (n_batch, len(driver_names)):
            raise ValueError(f"driver and driver_names must have the "
                             f"same length ({drivers.shape[-1]} != {len(driver_names)}")

        if bv_measurements.shape[1] < const.max_alt:
            # put nan placeholder for missing measurements if none provided
            missing_alts = np.zeros((n_batch, const.max_alt - n_alt_start, n_bv)) * np.nan
            bvs = np.concatenate((bv_measurements, missing_alts), axis=1)
        else:
            bvs = bv_measurements.copy()

        # z scale inputs and place into tensors
        driver_feat = trans.scale_driver(drivers, driver_names=driver_names)
        driver_feat = torch.tensor(driver_feat, dtype=torch.float)
        bv_feat, valid_mask = trans.scale_bv(bvs, 'radar')
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
        if mean_replace_drs is not None and len(mean_replace_drs) > 0:
            missing_dr_idxs = np.hstack([const.dr_feat_map[n] for n in mean_replace_drs])
        else:
            missing_dr_idxs = np.array([])

        # Load hfp GAN
        if model_dir is None:
            hfp_gen, _ = persist.recreate(name=hfp_model)
        else:
            hfp_gen, _ = persist.recreate(name=hfp_model, dir_path=model_dir)
        hfp_gen.to(device)
        hfp_gen.eval()

        # Generate waves
        for i in tqdm(range(len(batch_idxs) - 1), desc='Generate HFP Batches', disable=disable1):
            dr_src = driver_feat[batch_idxs[i]:batch_idxs[i + 1]].to(device)
            gbv_src = bv_feat[batch_idxs[i]:batch_idxs[i + 1]].to(device)
            hfp_tgt = hfp_feats[batch_idxs[i]:batch_idxs[i + 1]].to(device)
            src_mask = alt_mask[batch_idxs[i]:batch_idxs[i + 1]].to(device)

            dr_src[:, missing_dr_idxs] = 0.0

            hfp_out, gb = hfp_gen(dr_src, gbv_src, hfp_tgt, ~src_mask)
            hfp_feats[batch_idxs[i]:batch_idxs[i + 1], ...] = hfp_out.detach().cpu()
            G_b[batch_idxs[i]:batch_idxs[i + 1]] = gb.detach().cpu()
        G_hfps_feats = hfp_feats.numpy()
        G_hfps = trans.get_hfp(hfp_feats.numpy())
        G_b = G_b.numpy()

        if return_z_scale:
            return G_hfps_feats, G_b
        else:
            return G_hfps, G_b


def generate_multi_bv(drivers: np.ndarray,
                      bvs: Union[None, np.ndarray] = None,
                      n_repeat: int = 10,
                      n_alt: int = const.max_alt,
                      bv_model: str ='bv_gan',
                      bv_type: str = 'radar',
                      cuda_index: Union[None, int] = None,
                      verbose: int = 1):
    """
    Generate multiple background variable profiles
    consistent with the historical distribution for each driver sample.
    Used for anomaly detection.

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
        number of bv profiles to generate for each driver sample
    n_alt: int, optional
        number of altitude measurements to draw, defaults to max_alt
    bv_model: str, optional
        name of bv GAN to use
    bv_type: str. optional
        name of the type of background variables to use (lidar or radar)
    cuda_index: int, optional
        GPU index to use when generating BVs and HFPs
    verbose: bool, optional
        set to show loading bar
    Returns
    -------------
    G_bvs: (n_samples x n_repeat x n_alt x n_bvs)
        output measurements at each requested altitude.
    """
    disable = bool(verbose < 1)
    G_bvs = np.zeros((len(drivers), n_repeat, const.max_alt, const.n_bv))

    for i in tqdm(range(len(drivers)), desc='Generating BV Samples', disable=disable):
        sampled_driver = drivers[[i], ...].repeat(n_repeat, 0)
        if bvs is not None:
            sampled_bv = bvs[[i], ...].repeat(n_repeat, 0)
        else:
            sampled_bv = None
        G_bvs[i, ...] = generate_bvs(sampled_driver,
                                     bv_measurements=sampled_bv,
                                     bv_model=bv_model,
                                     bv_type=bv_type,
                                     cuda_index=cuda_index,
                                     verbose=False)
    return G_bvs


def generate_multi_hfp(drivers: np.ndarray,
                       bvs: np.ndarray,
                       n_repeat: int = 10,
                       hfp_model: str = 'hfp_gan',
                       cuda_index: Union[None, int] = None,
                       verbose: int = 1):
    """
    Generate  HFP waves consistent with the historical distribution for each driver sample.
    Creates multiple samples for a single input driver/bv profile to be using in anomaly detection

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    bvs: np.ndarray, optional
        n_samples x n_alt_in x n_meas input list of altitude measurements,
        n_alt_in should be less than n_alt but greater than zero. These represent fixed
        measurements that the HFP GAN will use as conditioning.
    n_repeat: int, optional
        number of hfp waves to generate for each driver sample
    hfp_model: str, optional
        name of hfp gan to use
    cuda_index: int, optional
        GPU index to use when generating and HFPs
    verbose: bool, optional
        set to show loading bar
    Returns
    -------------
    samples: (np.ndarray, np.ndarray)
        1) G_hfps (n_samples x n_repeat x 1 x n_hfps) generated hfp waves
        2) G_b (n_sample x n_repeat) probabilities that the generated wave is present
    """
    disable = bool(verbose < 1)
    G_hfps = np.zeros((len(drivers), n_repeat, const.n_waves, const.n_hfp))
    G_b = np.zeros((len(drivers), n_repeat))

    for i in tqdm(range(len(drivers)), desc='Generating HFP Samples', disable=disable):
        sampled_driver = drivers[[i], ...].repeat(n_repeat, 0)
        sampled_bv = bvs[[i], ...].repeat(n_repeat, 0)
        G_hfps[i, ...], G_b[i, ...] = generate_hfps(sampled_driver,
                                                    bv_measurements=sampled_bv,
                                                    hfp_model=hfp_model,
                                                    cuda_index=cuda_index,
                                                    verbose=False)
    return G_hfps, G_b


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


def hellinger_scores_bv(real: np.ndarray,
                        fake: np.ndarray,
                        mask: Union[None, np.ndarray]=None,
                        bins: Union[None, int]=None,
                        filter_length: Union[None, int]=None,
                        return_hist_info: bool=False,
                        z_scale: bool=True,
                        z_scale_input: bool=False,
                        bv_type: str='radar'):
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


def hellinger_scores_wtec(real: np.ndarray,
                          fake: np.ndarray,
                          n_bins: Union[None, int] = None,
                          filter_length: Union[None, int] = None,
                          z_scale: bool = True,
                          z_scale_inputs: bool = False,
                          dataset_name: Union[None, str] = const.wtec_default_dataset,
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
    dataset_name: str
        specify dataset type for z-scaling
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
    r_hists = np.zeros((n_bins, real.shape[1]))
    f_hists = np.zeros((n_bins, real.shape[1]))
    edges = np.zeros((n_bins + 1, real.shape[1]))

    for i in range(real.shape[1]):
        if z_scale:
            if z_scale_inputs:
                r = real[:, i]
                f = fake[:, i]
            else:
                r = trans.scale_wtec(real, dataset_name=dataset_name)[0][:, i]
                f = trans.scale_wtec(fake, dataset_name=dataset_name)[0][:, i]
            binranges = (max(const.wtec_zscale_dict[dataset_name]['z_ranges'][i][0], min(np.nanmin(r), np.nanmin(f))),
                         min(const.wtec_zscale_dict[dataset_name]['z_ranges'][i][1], max(np.nanmax(r), np.nanmax(f))))
            args = {'bins': n_bins, 'range': binranges, 'density': True}
        else:
            if z_scale_inputs:
                r = trans.get_wtec(real, dataset_name=dataset_name)[:, i]
                f = trans.get_wtec(fake, dataset_name=dataset_name)[:, i]
            else:
                r = real[:, i]
                f = fake[:, i]
            binranges = (max(const.wtec_zscale_dict[dataset_name]['meas_ranges'][i][0], min(np.nanmin(r), np.nanmin(f))),
                         min(const.wtec_zscale_dict[dataset_name]['meas_ranges'][i][1], max(np.nanmax(r), np.nanmax(f))))
            args = {'bins': n_bins, 'range': binranges, 'density': True}
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


def load_h5_data(fname,
                 bv_type: str = 'radar',
                 load_hfp: bool = False,
                 n_samples: Union[None, int] = None,
                 random_start: bool = False,
                 driver_names=const.driver_names):
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
    random_start: bool. optional
        randomize starting index to select n_samples from
    driver_names: list. optional
        list of driver names to load
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
                            for driver_name in driver_names
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
    drivers = drivers[valid_mask]
    bvs = bvs[valid_mask]
    unix_time = unix_time[valid_mask]

    # Get altitude mask for bvs and wave mask for hfps
    _, alt_mask = mask_altitude(torch.tensor(bvs, dtype=torch.float))
    alt_mask = alt_mask.detach().numpy()

    # Select starting index and samples to return
    if n_samples is None or n_samples > valid_mask.sum():
        n_samples = valid_mask.sum()
    if random_start and n_samples and len(drivers) - n_samples > 0:
        start_index = np.random.randint(low=0, high=len(drivers) - n_samples)
    else:
        start_index = 0
    drivers = drivers[start_index:start_index+n_samples]
    bvs = bvs[start_index:start_index+n_samples]
    alt_mask = alt_mask[start_index:start_index+n_samples]
    unix_time = unix_time[start_index:start_index+n_samples]

    # If set return valid hfps and wave mask
    if load_hfp:
        hfps = hfps[valid_mask]
        wave_mask = ~(np.isnan(hfps).any(-1))
        hfps = hfps[start_index:start_index+n_samples]
        wave_mask = wave_mask[start_index:start_index+n_samples]
        return drivers, bvs, alt_mask, hfps, wave_mask, unix_time

    else:
        return drivers, bvs, alt_mask, unix_time


def load_wtec_h5(fname: str,
                 dataset_name=const.wtec_default_dataset,
                 n_samples: Union[None, int] = None,
                 avg_coefficients: Union[None, List[float]] = None,
                 random_start: bool = True,):
    """
    loads and returns external drivers, tec wave parameters,
    and unix timestamp all aligned in time with outlier/invalid
    data filtered out

    Parameters
    -------------
    fname: str
        name of h5 file to load the data from
    dataset_name: str
        specify dataset type for z-scaling
    n_samples: int. optional
        number of samples to load (None to load all samples)
    avg_coefficients: list (n_wtec_feat,)
        z-scaled averaging coefficients to smooth out
        the original tec wave parameter distributions.
    random_start: bool. optional
        randomize starting index to select n_samples from
    Returns
    -------------
    drivers: np.ndarray
        (n_samples x n_wtec_dr) external drivers.
    wtec: np.ndarray
        (n_samples x n_wtec) tec wave parameter samples.
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
        # Read tec wave parameters form h5 file
        # --------------------------------------
        wtec_dict = f['TEC_Waves']
        wtec = np.stack([wtec_dict[wtec_name]
                         for wtec_name in const.wtec_names
                         if wtec_name in wtec_dict.keys()], axis=-1)

    # Get valid bvs and altitude mask
    valid_wtec_mask = compute_valid_wtec(wtec)
    valid_mask = valid_wtec_mask & ~(np.isnan(drivers).any(-1))

    # Filter out any invalid samples
    drivers = drivers[valid_mask]
    wtec = wtec[valid_mask]
    unix_time = unix_time[valid_mask]

    if n_samples is None or n_samples > valid_mask.sum():
        n_samples = valid_mask.sum()
    if random_start and n_samples and len(drivers) - n_samples > 0:
        start_index = np.random.randint(low=0, high=len(drivers) - n_samples)
    else:
        start_index = 0
    drivers = drivers[start_index:start_index+n_samples]
    wtec = wtec[start_index:start_index+n_samples]
    unix_time = unix_time[start_index:start_index+n_samples]

    # Perform averaging on tec wave data
    if avg_coefficients is not None:
        wtec = average_wtec(wtec,
                            dataset_name=dataset_name,
                            avg_coefficients=avg_coefficients,
                            z_scale_input=False)

    return drivers, wtec, unix_time


def average_wtec(wtec: np.ndarray,
                 avg_coefficients: List[float] = const.wtec_avg_coefficients,
                 z_scale_input: bool = False,
                 dataset_name: Union[None, str] = const.wtec_default_dataset):
    """
    loads and returns external drivers, tec wave parameters,
    and unix timestamp all aligned in time with outlier/invalid
    data filtered out

    Parameters
    -------------
    wtec: np.ndarray
        tec wave parameter data
    avg_coefficients: list (n_wtec_feat,)
        z-scaled averaging coefficients to smooth out
        the original tec wave parameter distributions.
    z_scale_input: bool
        set if the input wtec data is already z-scaled
    dataset_name: str
        specify dataset type for z-scaling
    Returns
    -------------
    wtec: np.ndarray
        (n_samples x n_wtec) tec wave parameter samples.
    """
    n, n_feat = wtec.shape[0], wtec.shape[1]
    if z_scale_input:
        averaged_wtec = wtec + np.random.randn(n, n_feat) * avg_coefficients
    else:
        averaged_wtec = wtec.copy()
        wtec_feat, valid_mask = trans.scale_wtec(wtec, dataset_name=dataset_name)
        averaged_wtec[valid_mask] = wtec_feat + np.random.randn(n, n_feat) * avg_coefficients
        averaged_wtec = trans.get_wtec(averaged_wtec, dataset_name=dataset_name)

    return averaged_wtec
