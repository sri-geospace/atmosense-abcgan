"""
Code for top level interface.

This code is added to the main package level in __init__.py
"""
import numpy as np
import abcgan.constants as const
import abcgan.transforms as trans
from abcgan import persist
from abcgan.mask import mask_altitude
import torch
import h5py
from scipy.special import logsumexp
from numpy import linalg as LA


def generate(drivers, driver_names=const.driver_names,
             measurements=None, n_alt=const.max_alt,
             model='mm_gan_radar', bv_type='radar'):
    """
    Generate synthetic data consistent with the historical distribution.

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
    model: str, optional
        name of model to use
    bv_type: str. optional
        name of the type of background variables to use (lidar or radar)

    Returns
    -------------
    samples: np.ndarray
        n_samples x n_alt x n_meas output measurements at each requested
        altitude. If measurements is not None then the measurements for
        the first n_alt_in will be copied over from the input.
    """
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

    # transform inputs
    driver_feat = trans.scale_driver(drivers, driver_names=driver_names)
    bv_feat, valid_mask = trans.scale_bv(measurements, bv_type)

    driver_feat = torch.tensor(driver_feat, dtype=torch.float)
    bv_feat = torch.tensor(bv_feat, dtype=torch.float)

    bv_feat, alt_mask = mask_altitude(bv_feat)

    # build the generator
    gen, crit = persist.recreate(name=model)
    if gen.transformer.dr_emb.shape[0] != driver_feat.shape[-1]:
        raise ValueError(f"Model must be trained with "
                         f"{driver_feat.shape[-1]} drivers.")

    with torch.no_grad():
        # iteratively build altitude profile
        for i_alt in range(n_alt_start, n_alt):
            bv_out = gen(driver_feat, bv_feat)
            # fill in next altitude
            bv_feat[:, i_alt, :] = bv_out[:, i_alt, :]
        samples = trans.get_bv(bv_feat[:, :n_alt, :].cpu().numpy(), bv_type)
    return samples


def discriminate(drivers, measurements,
                 driver_names=const.driver_names, model='mm_gan_radar',
                 bv_type='radar'):
    """
    Score how well the measurements match with historical observations.

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    driver_names: list
        list of names of driving parameters
    measurements: np.ndarray
        n_samples x n_alt_in x n_meas input list of altitude measurements,
        n_alt_in should be less than max_alt.
    model: str, optional
        name of model to use
    bv_type: str. optional
        name of the type of background variables to use (lidar or radar)

    Returns
    -------------
    scores: np.ndarray
        n_samples x n_alt output normalcy scores in the range [0, 1.0].
    """

    n_batch, n_alt = measurements.shape[:2]
    gen, crit = persist.recreate(name=model)

    driver_feat = trans.scale_driver(drivers, driver_names)
    bv_feat, valid_mask = trans.scale_bv(measurements, bv_type)

    if crit.transformer.dr_emb.shape[0] != driver_feat.shape[-1]:
        raise ValueError(f"Model must be trained with "
                         f"{driver_feat.shape[-1]} drivers.")

    driver_feat = torch.tensor(driver_feat, dtype=torch.float)
    bv_feat = torch.tensor(bv_feat, dtype=torch.float)

    bv_feat, alt_mask = mask_altitude(bv_feat)

    with torch.no_grad():
        scores = crit(bv_feat, driver_feat, bv_feat, ~alt_mask)
        scores = scores.view(n_batch, -1)[:, :n_alt].cpu().numpy()
    return scores


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

    with torch.no_grad():
        predicted_feats = dr_gen(driver_feats)

    predicted_drivers = trans.get_driver(predicted_feats.cpu().numpy())
    return predicted_drivers


def gen_stats(drivers, data=None, model='mm_gan_radar', bv_type='radar'):
    """
    Statistical distribution of 10,000 upper altitude data points conditioned on driver parameters and lower altitude measurements.

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    data: np.ndarray
        n_samples x n_alt_in x n_meas
    model: str, optional
        name of model to use
    bv_type: str. optional
        name of the type of background variables to use (lidar or radar)

    Returns
    -------------
    samples: [np.ndarray, np.ndarray]
        2xn_avg*n_samples x n_alt x n_feat output anomaly scores (unbounded). The first element is the fake output.
        The second array entry contains the scaled background variables with repeats
    """
    n_avg = 10000
    # transform inputs
    driver_feat = trans.scale_driver(drivers)
    bv_feat, valid_mask = trans.scale_bv(data, bv_type)

    driver_feat = torch.tensor(driver_feat, dtype=torch.float)
    bv_feat = torch.tensor(bv_feat, dtype=torch.float)
    bv_feat, alt_mask = mask_altitude(bv_feat)

    # build the generator
    gen, crit = persist.recreate(name=model)

    # expand the driver and bv
    with torch.no_grad():
        a = driver_feat.repeat(n_avg, 1)
        data = bv_feat.repeat(n_avg, 1, 1)
        fakes = gen(a, data)

    samples = [fakes, data]
    return samples


def anomaly_estimation_1d(fakes, data):
    """
        compute an unbounded anomaly score for a new data sample using logsumexp computation method

        Parameters
        -------------
        fakes: torch.Tensor
            n_samples x n_alt x n_features background variables
        data: torch.Tensor
            1 x n_alt x n_features broadcast n_samples times to match fakes data shape

        Returns
        -------------
        anomalies: 1 xnp.ndarray, np.ndarray
            1 x n_alt x n_feat output of anomaly scores (unbounded).
        """

    # data scatter to bound the 1-sphere
    std = fakes.std(axis=0)
    std = std.repeat(fakes.shape[0], 1, 1).numpy()
    # number of samples
    [n, _, _] = fakes.shape
    # 'homogeneous' linewidth (decreases to match the number of samples)
    # sigma decreases with more samples.
    sigma = 2 * std / n

    a = (fakes.numpy() - data.numpy()) / sigma

    # anomaly score is not bounded
    anomalies = logsumexp(-0.5 * a ** 2, b=1 / sigma * np.power(np.pi * 2, -1 / 2) * np.ones(a.shape),
                          axis=0)
    return anomalies


def anomaly_estimation_nd(fakes, data):
    """
        compute an unbounded anomaly score for a new data sample using logsumexp computation method (N-dimensional)

        Parameters
        -------------
        fakes: torch.Tensor
            n_samples x n_alt x n_features background variables
        data: torch.Tensor
            1 x n_alt x n_features broadcast n_samples times to match fakes data shape

        Returns
        -------------
        anomalies: 1 xnp.ndarray, np.ndarray
            1 x n_alt x n_feat output of anomaly scores (unbounded).
        """

    # data scatter to bound the 1-sphere
    s = fakes.std(axis=0)
    s = s.numpy()
    print("s.shape = {}".format(s.shape))
    # number of samples
    [n, altitude, feat] = fakes.shape
    std = np.zeros([1, altitude])

    # use a geometric mean as a stand in for a full multivariate normal distribution.
    for a in range(altitude):
        tmp = 1
        for f in range(feat):
            tmp *= s[a, f]
        std[0, a] = np.power(tmp, 1 / feat)

    print("std shape = {}".format(std.shape))

    # 'homogeneous' linewidth (decreases to match the number of samples)
    # sigma decreases with more samples.
    sigma = 2 * std / n
    print("sigma (before repeat) = {}".format(sigma.shape))
    sigma = np.repeat(sigma, n, axis=0)
    print("sigma (after repeat) = {}".format(sigma.shape))

    a = LA.norm((fakes.numpy() - data.numpy()), axis=2)
    print("a = {}".format(a.shape))
    # anomaly score is not bounded
    anomalies = logsumexp(-0.5 * a ** 2, b=1 / sigma * np.power(np.pi * 2, -1 / 2) * np.ones(a.shape), axis=0)
    return anomalies


def anomaly_score(drivers, data=None, model='mm_gan_radar', bv_type='radar'):
    """
    returns unbounded anomaly score for a given set of driver parameters and data. more positive numbers are more confident.

    Parameters
    -------------
    drivers: np.ndarray
        1 x n_drivers input driving parameters (not z-scaled). one sample at a time
    data: np.ndarray
        1 x n_alt_in x n_meas
    model: str, optional
        name of model to use
    bv_type: str. optional
        name of the type of background variables to use (lidar or radar)

    Returns
    -------------
    anomalies: 1 xnp.ndarray, np.ndarray
        1 x n_alt x n_feat output of anomaly scores (unbounded).
    """
    [fakes, data] = gen_stats(drivers, data, model, bv_type)

    return anomaly_estimation_nd(fakes, data)


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
