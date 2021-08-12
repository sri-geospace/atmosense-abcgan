"""
Code for top level interface to the saved model.

"""
import numpy as np
from . import constants as const
from . import transforms as trans
from . import persist
from .mask import mask_altitude
import torch
import h5py

max_alt = const.max_alt


def generate(drivers, measurements=None, n_alt=max_alt, model='wgan_gp_v7'):
    """
    Generate synthetic data consistent with the historical distribution.

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    measurements: np.ndarray, optional
        n_samples x n_alt_in x n_meas input list of altitude measurements,
        n_alt_in should be less than n_alt. These represent fixed
        measurements for the lowest altitudes to condition on.
        Usually left as default (None)
    n_alt: int, optional
        number of altitude measurements to draw, defaults to max_alt
    model: str, optional
        name of model to use

    Returns
    -------------
    samples: np.ndarray
        n_samples x n_alt x n_meas output measurements at each requested
        altitude. If measurements is not None then the measurements for
        the first n_alt_in will be copied over from the input.
    """
    if n_alt > max_alt:
        raise ValueError(f"Requested {n_alt} altitudes but only {max_alt}"
                         f" can be simulated.")
    if measurements is None:
        # put placeholder measurements if none provided
        measurements = np.zeros((drivers.shape[0], 0, const.n_bv))

    n_batch = drivers.shape[0]
    n_alt_start = measurements.shape[1]

    # verify the correct shapes for the inputs
    if drivers.shape != (n_batch, const.n_driver):
        raise ValueError(f"Drivers must have shape n_batch x {const.n_driver}")
    if measurements.shape != (n_batch, n_alt_start, const.n_bv):
        raise ValueError(f"Measurement shape must be n_batch x "
                         f"{n_alt_start} x {const.n_bv}.")

    # transform inputs
    driver_feat = trans.scale_driver(drivers)
    bv_feat, valid_mask = trans.scale_bv(measurements)

    driver_feat = torch.tensor(driver_feat, dtype=torch.float)
    bv_feat = torch.tensor(bv_feat, dtype=torch.float)

    bv_feat, alt_mask = mask_altitude(bv_feat)

    # build the generator
    gen, crit = persist.recreate(name=model)

    with torch.no_grad():
        # iteratively build altitude profile
        for i_alt in range(n_alt_start, n_alt):
            bv_out = gen(driver_feat, bv_feat)
            # fill in next altitude
            bv_feat[:, i_alt, :] = bv_out[:, i_alt, :]
        samples = trans.get_bv(bv_feat[:, :n_alt, :].cpu().numpy())
    return samples


def discriminate(drivers, measurements, model='wgan_gp_v7'):
    """
    Score how well the measurements match with historical observations.

    Parameters
    -------------
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    measurements: np.ndarray
        n_samples x n_alt_in x n_meas input list of altitude measurements,
        n_alt_in should be less than max_alt.
    model: str, optional
        name of model to use

    Returns
    -------------
    scores: np.ndarray
        n_samples x n_alt output normalcy scores in the range [0, 1.0].
    """

    n_batch, n_alt = measurements.shape[:2]
    gen, crit = persist.recreate(name=model)
    driver_feat = trans.scale_driver(drivers)
    bv_feat, valid_mask = trans.scale_bv(measurements)

    driver_feat = torch.tensor(driver_feat, dtype=torch.float)
    bv_feat = torch.tensor(bv_feat, dtype=torch.float)

    bv_feat, alt_mask = mask_altitude(bv_feat)

    with torch.no_grad():
        scores = crit(bv_feat, driver_feat, bv_feat, ~alt_mask)
        scores = scores.view(n_batch, -1)[:, :n_alt].cpu().numpy()
    return scores


def stack_drivers(driver_dict):
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
    return np.stack([driver_dict[k] for k in const.driver_names],
                    axis=-1)


def stack_bvs(bv_dict):
    """
    Stacks drivers in appropriate format.

    This function is provided for convenience.

    Parameters
    ----------------
    bv_dict: dict
        Dictionary mapping names of background variables
        to numpy arrays with values for those bvs. Each
        array should have shape n_samples x n_altitudes.
        Can also use `h5py.Group`.

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
    return np.stack([bv_dict[k] for k in const.bv_names],
                    axis=-1)
