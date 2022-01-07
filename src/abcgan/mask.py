import torch
import numpy as np
import abcgan.constants as const


def prev_driver_mask(unix_time):
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

    # Determine time differences between each
    # sample and the previous 6 samples
    diff = np.zeros((len(unix_time), 6))
    for i in range(6, len(unix_time)):
        diff[i, :] = np.flip(unix_time[i] - unix_time[i - 6: i])

    # Create mask of samples that have a previous sample equal to
    # the specified delay time and a mapping vector to the
    # current driver data structure
    dr_mask = (diff == const.dr_delay).any(-1)
    prev_dr_map = np.where(diff == const.dr_delay)[0] - \
                  np.where(diff == const.dr_delay)[1]

    return prev_dr_map, dr_mask


def mask_altitude(bv_feat):
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
