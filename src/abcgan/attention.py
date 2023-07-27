import torch
import numpy as np
from typing import List

import abcgan.constants as const
import abcgan.transforms as trans
from abcgan import persist
from abcgan.mask import mask_altitude


def get_encoder_attn(layers, src, src_mask, src_key_mask):
    """
    Collects attention masks from each encoder layer in the transformer

    Parameters
    -------------
    layers: list
        list of pytorch encoder layers from transformer
    src: torch.tensor
        embedded source tensor
    src_mask: torch.tensor
        constant source key mask from transformer
    src_key_mask: torch.tensor
        mask for missing source data

    Returns
    -------------
    samples: torch.tensor
        n_layers x n_samples x n_alt x n_alts output of attention mask or weights
        for each encoder layer in the transformer
    """
    output = src.clone()
    attention_masks = torch.zeros((len(layers), src.shape[1], src.shape[0], src.shape[0]))

    s2, att_mask = layers[0].self_attn(output, output, output, attn_mask=src_mask,
                                       key_padding_mask=src_key_mask)
    attention_masks[0, ...] = att_mask
    s1 = src + layers[0].dropout1(s2)
    s1 = layers[0].norm1(s1)
    s2 = layers[0].linear2(layers[0].dropout(layers[0].activation(layers[0].linear1(s1))))
    s1 = s1 + layers[0].dropout2(s2)
    s1 = layers[0].norm2(s1)
    for i in range(1, len(layers)):
        s2, att_mask = layers[i].self_attn(s1, s1, s1, attn_mask=src_mask,
                                           key_padding_mask=src_key_mask)
        attention_masks[i, ...] = att_mask
        s1 = s1 + layers[i].dropout1(s2)
        s1 = layers[i].norm1(s1)
        s2 = layers[i].linear2(layers[i].dropout(layers[i].activation(layers[i].linear1(s1))))
        s1 = s1 + layers[i].dropout2(s2)
        s1 = layers[i].norm2(s1)

    return attention_masks


def get_decoder_attn(transformer, tgt, memory, key_padding_mask):

    tgt = tgt.clone()
    layers = transformer.decoder.layers
    attn_map = torch.zeros(len(layers), tgt.shape[1],
                           const.n_waves, memory.shape[0])

    for i in range(len(layers)):
        tgt2 = layers[i].self_attn(tgt, tgt, tgt, attn_mask=transformer.tgt_mask)[0]
        tgt = tgt + layers[i].dropout1(tgt2)
        tgt = layers[i].norm1(tgt)
        tgt2, attn_map[i, ...] = layers[i].multihead_attn(tgt, memory, memory,
                                                          key_padding_mask=key_padding_mask)
        tgt = tgt + layers[i].dropout2(tgt2)
        tgt = layers[i].norm2(tgt)
        tgt2 = layers[i].linear2(layers[i].dropout(layers[i].activation(layers[i].linear1(tgt))))
        tgt = tgt + layers[i].dropout3(tgt2)
        tgt = layers[i].norm3(tgt)

    return attn_map


def collect_hfp_attn_map(drivers: np.ndarray,
                         bvs: np.ndarray,
                         driver_names: List[str] = const.driver_names,
                         model: str = 'hfp_gan',
                         bv_type: str = 'radar'):
    """
    function to collect attention map from the HFP GAN's decoder

    Parameters
    -------------
    bvs: np.ndarray
        n_samples x n_alt x n_feat (not z-scaled)
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    driver_names: list
        list of names of driving parameters
    model: str, optional
        name of model to use
    bv_type: str. optional
        name of the type of background variables to use (lidar or radar)

    Returns
    -------------
    samples: np.ndarray
        n_waves x n_alt map found in transformer
        altitude.
    """
    n_batch = drivers.shape[0]

    # verify the correct shapes for the inputs
    if drivers.shape != (n_batch, len(driver_names)):
        raise ValueError(f"driver and driver_names must have the "
                         f"same length ({drivers.shape[-1]} != {len(driver_names)}")

    # transform inputs
    driver_feat = trans.scale_driver(drivers, driver_names=const.driver_names)
    bv_feat, valid_mask = trans.scale_bv(bvs, bv_type)

    driver_feat = torch.tensor(driver_feat, dtype=torch.float)
    bv_feat = torch.tensor(bv_feat, dtype=torch.float)

    bv_feat, alt_mask = mask_altitude(bv_feat)
    hfp_gan, _ = persist.recreate(name=model)
    hfp_gan.eval()
    transformer = hfp_gan.transformer
    src_key_mask = ~alt_mask

    with torch.no_grad():
        bv_shift = torch.roll(bv_feat.transpose(0, 1) @ transformer.bv_emb, 1, dims=0)
        bv_shift[0, :, :] = transformer.bv_start_token.unsqueeze(1)
        src = (driver_feat.unsqueeze(0) @ transformer.dr_emb + bv_shift + transformer.bv_position.unsqueeze(1))
        tgt = (driver_feat.unsqueeze(0) @ transformer.dr_emb + transformer.hfp_start_token.unsqueeze(1))
        memory = transformer.encoder(src, mask=transformer.src_mask,
                                     src_key_padding_mask=src_key_mask.to(transformer.src_mask.dtype))
        attn_map = get_decoder_attn(transformer, tgt, memory, src_key_mask.to(transformer.src_mask.dtype))

    attn_map = torch.mean(attn_map, dim=0).detach().numpy()
    return attn_map


def collect_bv_attn_map(drivers: np.ndarray,
                        bvs: np.ndarray,
                        driver_names: List[str] = const.driver_names,
                        model: str = 'bv_gan',
                        bv_type: str = 'radar'):
    """
    function to collect attention map from weights in pre-trained model

    Parameters
    -------------
    bvs: np.ndarray
        n_samples x n_alt x n_feat (not z-scaled)
    drivers: np.ndarray
        n_samples x n_drivers input list of driving parameters (not z-scaled).
    driver_names: list
        list of names of driving parameters
    model: str, optional
        name of model to use
    bv_type: str. optional
        name of the type of background variables to use (lidar or radar)

    Returns
    -------------
    samples: np.ndarray
        n_alt x n_alt atten map found in transformer
        altitude.
    """
    n_batch = drivers.shape[0]

    # verify the correct shapes for the inputs
    if drivers.shape != (n_batch, len(driver_names)):
        raise ValueError(f"driver and driver_names must have the "
                         f"same length ({drivers.shape[-1]} != {len(driver_names)}")

    # transform inputs
    driver_feat = trans.scale_driver(drivers, driver_names=const.driver_names)
    bv_feat, valid_mask = trans.scale_bv(bvs, bv_type)

    driver_feat = torch.tensor(driver_feat, dtype=torch.float)
    bv_feat = torch.tensor(bv_feat, dtype=torch.float)

    bv_feat, alt_mask = mask_altitude(bv_feat)
    gan, _ = persist.recreate(name=model)
    gan.eval()
    transformer = gan.transformer
    src_key_mask = ~alt_mask

    with torch.no_grad():
        bv_shift = torch.roll(bv_feat.transpose(0, 1) @ transformer.bv_emb, 1, dims=0)
        bv_shift[0, :, :] = transformer.start_token.unsqueeze(1)
        src = (driver_feat.unsqueeze(0) @ transformer.dr_emb + bv_shift + transformer.position.unsqueeze(1))
        encoder = transformer.encoder

        attn_output_weights = get_encoder_attn(encoder.layers, src,
                                               transformer.src_mask,
                                               src_key_mask.to(transformer.src_mask.dtype))

    attn_map = torch.mean(attn_output_weights, dim=0)
    return attn_map.detach().numpy()
