"""
Code for masking bad training data points. It replaces NaN values floating point numbers.

"""

import torch
import numpy as np


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
