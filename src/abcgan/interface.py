"""
Code for top level interface.

This code is added to the main package level in __init__.py
"""
import numpy as np
import os
import glob
import torch
import h5py
from tqdm import tqdm
from warnings import warn
from typing import List, Union, Tuple

import abcgan.constants as const
import abcgan.transforms as trans
from abcgan import persist
from abcgan.persist import dir_path as default_model_dir
from abcgan.mask import mask_altitude, context_mapping
from abcgan.transforms import compute_valid, compute_valid_hfp


def generate_wtec(drivers,
                  driver_names: list = const.wtec_dr_names,
                  mean_replace_drs: Union[None, List[str]] = None,
                  ctx_wtecs: Union[np.ndarray, None] = None,
                  tid_type: str = const.wtec_default_tid_type,
                  location: str = const.wtec_default_location,
                  model_name: Union[None, str] = None,
                  model_dir: Union[None, str] = None,
                  return_z_scale: bool = False,
                  batch_size: Union[int, None] = None,
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
    ctx_wtecs: np.ndarray. optional
        the wtec context samples to use when conditioning the GAN
    return_z_scale: bool, optional
        set to have the function return z scaled feature data
    tid_type: str
        specify dataset type for z-scaling
    location: str
        specify the location of the trained model
    model_name: str, optional
        name of WTEC GAN to use
    model_dir: str, optional
        directory to load model from
    batch_size: int, optional
        batch size to use
    cuda_index: int, optional
        GPU index to use when generating BVs and HFPs
    verbose: bool, optional
        set to show loading bar
    Returns
    -------------
    samples: np.ndarray
        1) n_samples x n_wtec output measurements
    """
    if model_dir is None:
        model_dir = default_model_dir

    if tid_type not in const.wtec_dict.keys():
        raise ValueError(f"{tid_type} is an invalid tid. Please use one of the following: "
                         f"{list(const.wtec_dict.keys())}")
    if f'{tid_type}_{location}' not in const.wtec_dataset_names:
        raise ValueError(f"{tid_type} at {location} is an invalid model. "
                         f"Please use one of the following combinations: "
                         f"{const.wtec_dataset_names}")

    if model_name is None:
        dataset_name = f'{tid_type}_{location}'
        model_name = f'wtec_gan_{dataset_name}' if ctx_wtecs is None else f'wtec_cgan_{dataset_name}'

    if model_dir is None and not os.path.exists(os.path.join(default_model_dir, f'{model_name}.pt')):
        raise ValueError(f"cannot find {model_name} model. Please use one of the following: "
                         f"{[os.path.basename(f)[:-3] for f in glob.glob(f'{default_model_dir}/wtec*.pt')]}")
    if model_dir is not None and not os.path.exists(os.path.join(model_dir, f'{model_name}.pt')):
        raise ValueError(f"cannot find {model_name} model in the following directory: {model_dir}")

    with torch.no_grad():
        disable1 = bool(verbose < 1)
        n_batch = drivers.shape[0]
        # verify the correct shapes for the inputs
        if drivers.shape != (n_batch, len(driver_names)):
            raise ValueError(f"driver and driver_names must have the "
                             f"same length ({drivers.shape[-1]} != {len(driver_names)}")

        # z scale wtec context if provided
        if ctx_wtecs is not None:
            n_context = ctx_wtecs.shape[-1] // const.n_wtec
            wtec_ctx_feat = ctx_wtecs.copy()
            for i in range(n_context):
                wtec_ctx_feat[:, i, :] = trans.scale_wtec(wtec_ctx_feat[:, i, :],
                                                          tid_type=tid_type)[0]
            wtec_ctx_feat = torch.tensor(wtec_ctx_feat, dtype=torch.float).flatten(1, 2)
        else:
            wtec_ctx_feat = torch.zeros((n_batch, 0), dtype=torch.float)

        # z scale inputs and place into tensors
        driver_feat = trans.scale_driver(drivers, driver_names=driver_names, data_type='wtec')
        driver_feat = torch.tensor(driver_feat, dtype=torch.float)
        driver_feat = torch.cat((driver_feat, wtec_ctx_feat), dim=-1)

        if torch.cuda.is_available() and cuda_index is not None:
            device = torch.device('cuda:' + str(cuda_index))
            if batch_size is None:
                if len(drivers) >= 500:
                    batch_size = 500
                else:
                    batch_size = len(drivers)
        else:
            device = torch.device('cpu')
            if batch_size is None:
                if len(drivers) >= 100:
                    batch_size = 100
                else:
                    batch_size = len(drivers)

        batch_idxs = np.hstack((np.arange(0, len(drivers), step=batch_size), [len(drivers)]))
        wtec_feats = torch.zeros((len(driver_feat), const.n_wtec))
        if mean_replace_drs is not None and len(mean_replace_drs) > 0:
            replace_dr_idxs = np.hstack([const.dr_feat_map[n] for n in mean_replace_drs])
        else:
            replace_dr_idxs = np.array([])

        # Load bv models
        if model_dir is None:
            wtec_gen, _ = persist.recreate(name=model_name)
        else:
            wtec_gen, _ = persist.recreate(name=model_name, dir_path=model_dir)

        if wtec_gen.cond_dim != driver_feat.shape[-1]:
            raise ValueError(f"Model must be trained with "
                             f"{driver_feat.shape[-1]} drivers.")
        wtec_gen.to(device)
        wtec_gen.eval()

        # iteratively build altitude profile
        for i in tqdm(range(len(batch_idxs) - 1),
                      desc=f'Generating {tid_type}_{location} Waves',
                      disable=disable1):
            dr_src = driver_feat[batch_idxs[i]:batch_idxs[i + 1]].to(device)
            dr_src[..., replace_dr_idxs] = 0.0
            wtec_feats[batch_idxs[i]:batch_idxs[i + 1], :] = wtec_gen(dr_src).detach().cpu()

        G_wtec_feats = wtec_feats.numpy()
        G_wtecs = trans.get_wtec(wtec_feats.numpy(), tid_type=tid_type)
        wtec_gen.cpu()

        if return_z_scale:
            return G_wtec_feats
        else:
            return G_wtecs


def generate_multi_wtec(drivers: np.ndarray,
                        ctx_wtecs: Union[np.ndarray, None] = None,
                        n_repeat: int = 10,
                        tid_type: str = const.wtec_default_tid_type,
                        location: str = const.wtec_default_location,
                        model_name: Union[None, str] = None,
                        model_dir: Union[None, str] = None,
                        cuda_index: Union[None, int] = None,
                        block_size: int = 40000,
                        batch_size: int = 5000,
                        verbose: int = 0):
    """
    Generate multiple background variable profiles and HFP waves
    consistent with the historical distribution for each driver sample

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    ctx_wtecs: np.ndarray. optional
        the wtec context samples to use when conditioning the GAN
    n_repeat: int, optional
        number of waves to generate for each driver sample
    tid_type: str
        specify dataset type for z-scaling
    location: str
        specify the location of the trained model
    model_name: str, optional
        name of WTEC GAN to use
    model_dir: str, optional
        directory to load model from
    block_size: int, optional
        first batch process size
    batch_size: int, optional
        second batch processing size
        (split twice block and batch to limit memory usage)
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

    block_idxs = np.hstack((np.arange(0, len(drivers), step=block_size // n_repeat), [len(drivers)]))
    for i in tqdm(range(len(block_idxs) - 1), desc=f'Sampling {tid_type}_{location} Waves', disable=disable):
        sampled_driver = drivers[block_idxs[i]:block_idxs[i + 1], ...].repeat(n_repeat, 0)
        if ctx_wtecs is not None:
            sampled_wtec_context = ctx_wtecs[block_idxs[i]:block_idxs[i + 1], ...].repeat(n_repeat, 0)
        else:
            sampled_wtec_context = None
        G_wtec[block_idxs[i]:block_idxs[i + 1], ...] = generate_wtec(sampled_driver,
                                                                     driver_names=const.wtec_dr_names,
                                                                     tid_type=tid_type,
                                                                     location=location,
                                                                     model_name=model_name,
                                                                     ctx_wtecs=sampled_wtec_context,
                                                                     model_dir=model_dir,
                                                                     cuda_index=cuda_index,
                                                                     batch_size=batch_size,
                                                                     verbose=False).reshape((-1, n_repeat, const.n_wtec))
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


def stack_drivers(driver_dict, driver_names=const.driver_names):
    """
    Stacks drivers in appropriate format. This function is provided for convenience.

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
    Stacks drivers in appropriate format. This function is provided for convenience.

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


def load_wtec_h5(fname: Union[None, str] = None,
                 locations: Union[list, str] = const.wtec_default_location,
                 tid_type: str = const.wtec_default_tid_type,
                 split: Union[None, str] = None,
                 n_samples: Union[None, int] = None,
                 n_context: int = 0,
                 context_padding: int = 1,
                 start_utc: Union[str, float, None] = None,
                 **kwargs):
    """
    loads and returns external drivers, tec wave parameters,
    and unix timestamp all aligned in time with outlier/invalid
    data filtered out

    Parameters
    -------------
    fname: str
        name of h5 file to load the data from. None will load tutorial subset
    tid_type: str
        specify dataset type for z-scaling
    locations: str
        site locations to load
    split: str. optional
        name of the dataset split to load ('train' or 'val' or None for all)
    n_context: int. optional
        the number of previous samples to load for each sample
        to be used as temporal context when conditioning the model
    context_padding: int. optional
        the number of previous samples to load for each sample
        to pad with
    n_samples: int. optional
        number of samples to load (None to load all samples)
    start_utc: bool. optional
        utc timestamp of the first sample to load. set to 'random'
        to get random start.
    Returns
    -------------
    data: dict
        dict of wtec data and metadata for each location. Each location includes:
            - "drivers": (n_samples x n_wtec_dr) external drivers.
            - "wtecs": (n_samples x n_wtec) tec wave parameter samples.
            - "ctx_wtecs": (n_samples x n_context, n_wtec) context tec wave parameter samples.
            - "wtecs": (n_samples, ) time stamp of each sample
            - "location": (str) location name
            - "tid_type": (str) data TID type
            - "dataset_name": (str) location name + TID type
    """
    if isinstance(locations, str):
        locations = [locations]

    if tid_type not in const.wtec_dict.keys():
        raise ValueError(f'{tid_type} is an invalid TID Type. Plz choose from the following:'
                         f' {list(const.wtec_dict.keys())}')
    for loc in locations:
        if loc not in const.wtec_sites.keys():
            raise ValueError(f'{loc} is unsupported. Plz choose from the following:'
                             f' {list(const.wtec_sites.keys())}')

    if fname is None:
        tutorial_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'tutorials')
        if os.path.exists(f'wtec_tutorial.h5'):
            fname = f'wtec_tutorial.h5'
        elif os.path.exists(os.path.join(tutorial_dir, f'wtec_tutorial.h5')):
            fname = os.path.join(tutorial_dir, f'wtec_tutorial.h5')
        else:
            raise ValueError(f"Unable to find default data file")
    elif not os.path.exists(fname):
        raise ValueError(f"Unable to find data file: {fname}")

    with h5py.File(fname, 'r') as hf:
        data = {loc: {} for loc in locations if tid_type in hf[loc].keys()}
        for loc in locations:
            if tid_type in hf[loc].keys():
                if split in ["train", "val"]:
                    split_indexes = hf[loc][tid_type][f"{split}_indexes"][:]
                else:
                    split_indexes = np.arange(len(hf[loc][tid_type]['UnixTime']))

                global_indexes = hf[loc][tid_type]["global_indexes"][:]
                dr_global = np.stack([hf['Global_Drivers'][name] for name in
                                      const.wtec_dr_names if name in
                                      hf['Global_Drivers'].keys()], axis=-1)[global_indexes]
                dr_global_names = [name for name in const.wtec_dr_names if name in hf['Global_Drivers'].keys()]
                dr_global = {name: dr_global[:, i] for i, name in enumerate(dr_global_names)}
                dr_all = {}
                for name in const.wtec_dr_names:
                    if name in dr_global.keys():
                        dr_all[name] = dr_global[name]
                    elif name in hf[loc][tid_type]["Drivers"]:
                        dr_all[name] = hf[loc][tid_type]["Drivers"][name]

                drs = np.stack([dr_all[n] for n in const.wtec_dr_names], axis=-1)[split_indexes]
                wtecs = np.stack([hf[loc][tid_type]["TEC_Waves"][name]
                                  for name in const.wtec_names if name in
                                  hf[loc][tid_type]["TEC_Waves"].keys()], axis=-1)[split_indexes]
                utc = hf[loc][tid_type]["UnixTime"][split_indexes]

                # Scale drivers and get valid  mask
                valid_mask = ~(np.isnan(drs).any(-1)) & trans.scale_wtec(wtecs, tid_type=tid_type)[1]
                if n_context > 0:
                    ctx_mapping, ctx_mask = context_mapping(utc[valid_mask],
                                                            n_context=n_context,
                                                            n_padding=context_padding)
                    data[loc]["drivers"] = drs[valid_mask & ctx_mask]
                    data[loc]["wtecs"] = wtecs[valid_mask & ctx_mask]
                    data[loc]["ctx_wtecs"] = wtecs[valid_mask][ctx_mapping]
                    data[loc]["utc"] = utc[valid_mask & ctx_mask]
                else:
                    data[loc]["drivers"] = drs[valid_mask]
                    data[loc]["wtecs"] = wtecs[valid_mask]
                    data[loc]["ctx_wtecs"] = np.zeros((wtecs[valid_mask].shape[0], 0, wtecs[valid_mask].shape[1]))
                    data[loc]["utc"] = utc[valid_mask]

                data[loc]["location"] = loc
                data[loc]["tid_type"] = tid_type
                data[loc]["dataset_name"] = f'{tid_type}_{loc}'
    hf.close()

    if start_utc == 'random':
        if n_samples is None:
            max_utc = np.min([data[loc]["utc"].max() for loc in locations])
        else:
            max_utc = np.min([data[loc]["utc"][-min(len(data[loc]["utc"]), n_samples):] for loc in locations])
        min_utc = np.max([data[loc]["utc"].min() for loc in locations])
        rand_start = np.random.uniform(low=min_utc, high=max_utc)
        start_indexes = [np.argmin(abs(rand_start - data[loc]['utc'])) for loc in locations]
    elif isinstance(start_utc, float):
        start_indexes = [np.argmin(abs(start_utc - data[loc]['utc'])) for loc in locations]
    else:
        start_indexes = [0 for loc in locations]

    for start, loc in zip(start_indexes, locations):
        if n_samples is None:
            end = len(data[loc]['utc'])
        else:
            end = min(start + n_samples, len(data[loc]['utc']))
        data[loc]["drivers"] = data[loc]["drivers"][start:end]
        data[loc]["wtecs"] = data[loc]["wtecs"][start:end]
        data[loc]["ctx_wtecs"] = data[loc]["ctx_wtecs"][start:end]
        data[loc]["utc"] = data[loc]["utc"][start:end]
    return data


def average_wtec(wtec: np.ndarray,
                 avg_coefficients: Union[np.ndarray, list, None] = None,
                 tid_type: Union[None, str] = const.wtec_default_tid_type,
                 z_scale_input: bool = False,):
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
    tid_type: str
        specify dataset type for z-scaling
    z_scale_input: bool
        set if the input wtec data is already z-scaled
    Returns
    -------------
    wtec: np.ndarray
        (n_samples x n_wtec) tec wave parameter samples.
    """
    n, n_feat = wtec.shape[0], wtec.shape[1]
    if avg_coefficients is None:
        if tid_type is None:
            raise ValueError('Must enter either averaging coefficients or dataset name')
        elif tid_type not in const.wtec_dict.keys():
            raise ValueError(f'{tid_type} is invalid. Must be on of the following: '
                             f'{list(const.wtec_dict.keys())}')
        avg_coefficients = const.wtec_dict[tid_type]['avg_coefficients']

    if z_scale_input:
        averaged_wtec = wtec + np.random.randn(n, n_feat) * avg_coefficients
    else:
        averaged_wtec = wtec.copy()
        wtec_feat, _ = trans.scale_wtec(wtec, tid_type=tid_type)
        averaged_wtec = wtec_feat + np.random.randn(n, n_feat) * avg_coefficients
        averaged_wtec = trans.get_wtec(averaged_wtec, tid_type=tid_type)

    return averaged_wtec
