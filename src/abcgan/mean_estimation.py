from typing import Optional
import torch
import torch.nn as nn
import abcgan.constants as const
from torch.nn.init import xavier_uniform_
from torch import Tensor


class Transformer(nn.Module):
    """
    Transformer with only the encoder

    Parameters
    -----------
    d_dr: int
        number of expected drivers features to condition with
    d_bv: int
        the number of expected bv features to input/output
    n_alt: int
        the number of bv altitude bins
    d_model: int
        the number of expected features in the encoder/decoder inputs
    nhead: int
        the number of heads in the multiheadattention models
    num_encoder_layers: int
        the number of sub-encoder-layers in the encoder
    dim_feedforward: int
        the dimension of the feedforward network model
    dropout: float
        the dropout value
    activation: str
        the activation function of encoder/decoder intermediate layer
    """

    def __init__(self,
                 d_dr: int = const.n_driver_feat,
                 d_bv: int = const.n_bv_feat,
                 n_alt: int = const.max_alt,
                 d_model: int = 64,
                 nhead: int = 1,
                 num_encoder_layers: int = 1,
                 dim_feedforward: int = 64,
                 dropout: float = 0.0,
                 activation: str = "relu") -> None:
        super().__init__()
        # args captured for persistence
        d_out = d_model
        self.input_args = {
            'd_dr': d_dr,
            'd_bv': d_bv,
            'n_alt': n_alt,
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'activation': activation,
        }

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        self.position = nn.Parameter(torch.zeros(n_alt, d_model))
        self.dr_emb = nn.Parameter(torch.zeros(d_dr, d_model))
        self.bv_emb = nn.Parameter(torch.zeros(d_bv, d_model))
        self.start_token = nn.Parameter(torch.zeros(1, d_model))

        self._reset_parameters()

        self.d_model = d_model
        self.d_bv = d_bv
        self.d_out = d_out
        self.nhead = nhead

        src_mask = self.generate_square_subsequent_mask(n_alt)
        self.register_buffer('src_mask', src_mask)

    def forward(self,
                driver_src: Tensor,
                bv_src: Tensor,
                src_key_padding_mask: Optional[Tensor] = None):
        """
        Take in and process masked source/target sequences.
        Args:
            driver_src: the sequence to the encoder (required).
            bv_src: the sequence to the decoder (required).
            src_key_padding_mask: the ByteTensor mask for src keys per batch
            (optional).
        Shape:
            - driver_src: :math:`(n_batch, d_dr)`.
            - bv_src: :math:`(n_batch, n_alt, d_bv)`.
            - src_key_padding_mask: :math:`(n_batch, n_alt)`.
        """
        bv_shift = torch.roll(bv_src.transpose(0, 1) @ self.bv_emb, 1, dims=0)
        bv_shift[0, :, :] = self.start_token.unsqueeze(1)
        src = (driver_src.unsqueeze(0) @ self.dr_emb + bv_shift +
               self.position.unsqueeze(1))
        key_mask = src_key_padding_mask.to(self.src_mask.dtype) if src_key_padding_mask is not None else None
        encoder_output = self.encoder(src, mask=self.src_mask,
                                      src_key_padding_mask=key_mask)
        est_bv = encoder_output.transpose(0, 1)
        est_bv[src_key_padding_mask] = 0.0
        return est_bv

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """
        Generate a square mask for the sequence. The masked positions are
        filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            self.position.normal_()
            self.start_token.normal_()
