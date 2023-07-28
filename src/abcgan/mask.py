import torch
import numpy as np
from typing import Union
import abcgan.constants as const


def prev_driver_mask(unix_time: np.ndarray):
    """
    Creates a driver mask of samples that have a previous sample
    and a mapping vector to the previous sample.

    Parameters
    ----------------
    unix_time: np.array
        time stamp of driver samples

    Returns
    ----------------
    prev_dr_map: np.array
        vector mapping each sample to its delayed sample
    dr_mask: torch.Tensor
        Mask of valid driver samples that have a delayed sample
    """

    dr_mask = np.zeros(len(unix_time), dtype=bool)
    prev_dr_map = []
    for i in range(len(unix_time)):
        diff = np.where(unix_time[i] - unix_time == const.dr_delay)[0]
        if len(diff) > 0:
            dr_mask[i] = True
            prev_dr_map.append(diff.item())
    prev_dr_map = np.array(prev_dr_map)

    return prev_dr_map, dr_mask


def context_mapping(unix_time: Union[list, np.ndarray],
                    n_context: int = 2,
                    n_padding: int = 1,
                    delay: int = 3600):
    """
    Creates a mask of samples that have a previous sample
    and a mapping vector to the previous sample.

    Parameters
    ----------------
    unix_time: np.array
        time stamp of samples
    n_context: int
        the number of previous samples to use as context
    n_padding: int
        the padding offset to begin collecting context samples
    delay: int
        delay in seconds between previous context samples

    Returns
    ----------------
    ctx_mapping: np.array
        vector mapping each sample to its delayed sample
    ctx_mask: np.array
        Mask of valid driver samples that have a delayed sample
    """

    ctx_mask = np.zeros(len(unix_time), dtype=bool)
    ctx_mapping = []
    delays = np.array([delay*(i + n_padding) for i in range(1, n_context + 1)])[::-1]
    for i in range(n_context + n_padding, len(unix_time)):
        c_start = i - n_context - n_padding
        c_stop = i - n_padding
        if (unix_time[i] - unix_time[c_start: c_stop] == delays).all():
            ctx_mapping.append(np.array([i - c - n_padding - 1 for c in range(n_context)])[::-1])
            ctx_mask[i] = True
        else:
            ctx_mask[i] = False
    ctx_mapping = np.array(ctx_mapping)

    return ctx_mapping, ctx_mask


def mask_altitude(bv_feat: torch.Tensor):
    """
    Creates an altitude mask for nans in bvs.

    Also replaces nans with numbers.

    Parameters
    ----------------
    bv_feat: torch.Tensor
        background variables

    Returns
    ----------------
    bv_feat: torch.Tensor
        bv_feat with nans replaced, done in place but returned for clarity
    alt_mask: torch.Tensor
        Mask that is true for valid altitudes

    Raises
    -----------------
    ValueError:
        If valid values are not contiguous.
    """
    squash = False
    if bv_feat.ndim == 2:
        bv_feat = bv_feat.unsqueeze(0)
        squash = True
    alt_mask = ~torch.isnan(bv_feat).any(-1)
    bv_feat[torch.isnan(bv_feat)] = 0.0
    # verify that alt_mask is valid
    chk_mask = torch.nn.functional.pad(alt_mask, (1, 1), value=False)
    start_cnt = (chk_mask[:, 1:] & ~chk_mask[:, :-1]).sum(-1)
    stop_cnt = (~chk_mask[:, 1:] & chk_mask[:, :-1]).sum(-1)
    valid = (((start_cnt == 1) & (stop_cnt == 1)) |
             ((start_cnt == 0) & (stop_cnt == 0)))
    if not valid.all():
        first = np.where(~valid.numpy())[0]
        raise ValueError(f"Invalid data more than one stop / start. "
                         f"First invalid row is: {first}.")
    if squash:
        bv_feat, alt_mask = bv_feat[0, ...], alt_mask[0, ...]
    return bv_feat, alt_mask
