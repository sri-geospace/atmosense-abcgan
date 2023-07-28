import torch
import torch.nn as nn
import abcgan.constants as const
from torch.nn.init import xavier_uniform_
from torch import Tensor
from typing import Optional
from abcgan.bv_model import Generator, Critic  # noqa


class HFP_Generator(nn.Module):
    """
    Generator Class

    Parameters
    -----------
    transformer: torch.nn.Module
        the transformer model object that estimates hfp features
    n_layers: int
        number of MLP layers
    latent_dim: int
        the dimension of the input latent vector
    img_dim: int
        the dimension of the images, fitted for the dataset used, a scalar
    hidden_dim: int
        the inner dimension, a scalar
    """

    def __init__(self,
                 transformer: nn.Module,
                 n_layers=4,
                 latent_dim=16,
                 img_dim=const.n_hfp_feat,
                 hidden_dim=128, ):
        super(HFP_Generator, self).__init__()

        cond_dim = transformer.d_model + latent_dim

        self.input_args = {
            'n_layers': n_layers,
            'latent_dim': latent_dim,
            'img_dim': img_dim,
            'hidden_dim': hidden_dim,
        }

        # Build the HFP neural network
        layers = [nn.Linear(cond_dim, hidden_dim),
                  nn.ReLU(inplace=True)]
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, img_dim))
        self.gen_model = nn.Sequential(*layers)

        self.transformer = transformer
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.img_dim = img_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self._reset_parameters()

    def forward(self, driver_src, bv_src, hfp_tgt,
                src_key_mask=None, tgt_key_mask=None, noise=None):
        """
        Function for completing a forward pass of the generator:
        Given a noise tensor,
        returns generated images.
        Parameters
        --------------
        driver_src: torch.Tensor
            tensor of driver features from data loader (n_batch, n_dr_feat)
        bv_src: torch.Tensor
            tensor of bv featrues from data loader (n_batch, n_alt, n_bv_feat)
        hfp_tgt: torch.Tensor
            tensor of hfp featrues from data loader (n_batch, n_wave, n_hfp_feat)
        src_key_mask: torch.Tensor, optional
            mask for bv features from data loader (n_alt, n_batch)
        tgt_key_mask: torch.Tensor, optional
            mask for hfp features from data loader (n_batch, n_wave)
        noise: torch.Tensor, optional
            a noise tensor with dimensions (n_batch, latent_dim)
        """
        if src_key_mask is None:
            src_key_mask = torch.zeros(
                bv_src.shape[0], bv_src.shape[1],
                dtype=torch.bool, device=bv_src.device)
        if tgt_key_mask is None:
            tgt_key_mask = torch.zeros(
                hfp_tgt.shape[0], hfp_tgt.shape[1],
                dtype=torch.bool, device=hfp_tgt.device)

        # Get Conditioning from transformer
        _, est_hfp = self.transformer(driver_src, bv_src, hfp_tgt, src_key_mask, tgt_key_mask)
        G_b = torch.sigmoid(est_hfp[..., -1]).flatten()
        est_hfp = est_hfp[..., :-1]

        if noise is None:
            noise = torch.randn(hfp_tgt.shape[0] * hfp_tgt.shape[1],
                                self.latent_dim,
                                dtype=driver_src.dtype,
                                device=driver_src.device)

        # Add conditioning and generate estimated hfps
        G_hfp = self.gen_model(torch.cat((est_hfp.flatten(0, 1), noise), 1))
        G_hfp = G_hfp.reshape(hfp_tgt.shape[0], hfp_tgt.shape[1], hfp_tgt.shape[2])
        return G_hfp, G_b

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)


class HFP_Critic(nn.Module):
    """
    Critic Class

    Parameters
    -------------
    transformer: torch.nn.Module
        transformer for the critic
    n_layers: int
        number of layers in MLP
    img_dim: int
        the dimension of the images, fitted for the dataset used, a scalar
    hidden_dim: int
        the inner dimension, a scalar
    """

    def __init__(self,
                 transformer: nn.Module,
                 n_layers=4,
                 img_dim=const.n_hfp_feat,
                 hidden_dim=128, ):
        super(HFP_Critic, self).__init__()
        cond_dim = transformer.d_model + img_dim

        self.input_args = {
            'n_layers': n_layers,
            'img_dim': img_dim,
            'hidden_dim': hidden_dim,
        }
        # Build the neural network
        layers = [nn.Linear(cond_dim, hidden_dim),
                  nn.ReLU(inplace=True)]
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, 1))
        self.crit_model = nn.Sequential(*layers)

        self.n_layers = n_layers
        self.img_dim = img_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.transformer = transformer
        self._reset_parameters()

    def forward(self, dr_src, real_bv, real_hfp, hfp_feat, src_key_mask=None, tgt_key_mask=None):
        """
        Critic forward

        Parameters
        ---------------
        dr_src: torch.Tensor
            tensor of driver features from data loader (n_batch, n_dr_feat)
        real_bv: torch.Tensor
            tensor of bv features from data loader (n_batch, n_alt, n_bv_feat)
        real_hfp: torch.Tensor
            tensor of real hfp features from data loader (n_batch, n_wave, n_bv_feat)
        hfp_feat: torch.Tensor
            tensor of hfp features from data loader (n_batch, n_wave, n_bv_feat)
        src_key_mask: torch.Tensor, optional
            mask for bv features from data loader (n_batch, n_alt)
        tgt_key_mask: torch.Tensor, optional
            mask for hfp features from data loader (n_batch, n_wave)
        """
        if src_key_mask is None:
            src_key_mask = torch.zeros(
                real_bv.shape[0], real_bv.shape[1],
                dtype=torch.bool, device=real_bv.device)
        if tgt_key_mask is None:
            tgt_key_mask = torch.zeros(
                real_hfp.shape[0], real_hfp.shape[1],
                dtype=torch.bool, device=real_hfp.device)
        _, est_hfp = self.transformer(dr_src, real_bv, real_hfp, src_key_mask, tgt_key_mask)

        cond_hfp = torch.cat((est_hfp.flatten(0, 1), hfp_feat.flatten(0, 1)), 1)
        hfp_pred = self.crit_model(cond_hfp)
        return hfp_pred

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)


class HFP_Transformer(nn.Module):
    """Transformer with only the encoder
    Parameters:
        d_model: the number of expected features in the encoder/decoder inputs
        nhead: the number of heads in the multiheadattention models
        num_encoder_layers: the number of sub-encoder-layers in the encoder
        dim_feedforward: the dimension of the feedforward network model
        dropout: the dropout value
        activation: the activation function of encoder/decoder intermediate layer
    """

    def __init__(self,
                 d_dr: int = const.n_driver_feat,
                 d_bv: int = const.n_bv_feat,
                 n_alt: int = const.max_alt,
                 n_waves: int = const.n_waves,
                 d_hfp: int = const.n_hfp_feat,
                 d_model: int = 256,
                 output_b: bool = False,
                 nhead: int = 1,
                 num_encoder_layers: int = 1,
                 num_decoder_layers: int = 1,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: str = "relu") -> None:
        super().__init__()
        # args captured for persistence
        self.input_args = {
            'd_dr': d_dr,
            'd_bv': d_bv,
            'n_alt': n_alt,
            'd_hfp': d_hfp,
            'n_waves': n_waves,
            'd_model': d_model,
            'output_b': output_b,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'activation': activation,
        }

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_encoder_layers, decoder_norm)

        self.bv_position = nn.Parameter(torch.zeros(n_alt, d_model))
        self.dr_emb = nn.Parameter(torch.zeros(d_dr, d_model))
        self.bv_emb = nn.Parameter(torch.zeros(d_bv, d_model))
        self.hfp_emb = nn.Parameter(torch.zeros(d_hfp, d_model))
        self.bv_start_token = nn.Parameter(torch.zeros(1, d_model))
        self.hfp_start_token = nn.Parameter(torch.zeros(1, d_model))
        if output_b:
            self.b_emb = nn.Parameter(torch.zeros(d_model, d_model + 1))

        self._reset_parameters()

        self.output_b = output_b
        self.d_model = d_model
        self.n_decode_layers = num_decoder_layers
        self.n_encode_layers = num_encoder_layers
        self.n_waves = n_waves
        self.d_bv = d_bv
        self.d_hfp = d_hfp
        self.nhead = nhead

        src_mask = self.generate_square_subsequent_mask(n_alt)
        self.register_buffer('src_mask', src_mask)
        tgt_mask = self.generate_square_subsequent_mask(n_waves)
        self.register_buffer('tgt_mask', tgt_mask)

    def forward(self,
                driver_src: Tensor,
                bv_src: Tensor,
                hfp_tgt: Tensor,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, ):
        """
        Take in and process masked source/target sequences.
        Args:
            driver_src: conditioning driver input (required).
            bv_src: the sequence to the encoder (required).
            hfp_tgt: the sequence to the decoder (required).
            src_key_padding_mask: the ByteTensor mask for src keys per batch
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch
            (optional).
        Shape:
            - driver_src: :math:`(n_batch, d_dr)`.
            - bv_src: :math:`(n_batch, n_alt, d_bv)`.
            - src_key_padding_mask: :math:`(n_batch, n_alt)`.
        """
        bv_shift = torch.roll(bv_src.transpose(0, 1) @ self.bv_emb, 1, dims=0)
        bv_shift[0, :, :] = self.bv_start_token.unsqueeze(1)
        src = (driver_src.unsqueeze(0) @ self.dr_emb + bv_shift +
               self.bv_position.unsqueeze(1))

        hfp_embedded = hfp_tgt.transpose(0, 1) @ self.hfp_emb
        hfp_shift = torch.roll(hfp_embedded, 1, dims=0)
        hfp_shift[0, :, :] = self.hfp_start_token.unsqueeze(1)
        tgt = (driver_src.unsqueeze(0) @ self.dr_emb + hfp_shift)

        key_mask = src_key_padding_mask.to(self.src_mask.dtype) if src_key_padding_mask is not None else None
        encoder_output = self.encoder(src, mask=self.src_mask,
                                      src_key_padding_mask=key_mask)
        decoder_output = self.decoder(tgt, encoder_output,
                                      tgt_mask=self.tgt_mask,
                                      memory_key_padding_mask=key_mask)

        if self.output_b:
            decoder_output = (decoder_output @ self.b_emb)

        bv_output = encoder_output.transpose(0, 1)
        hfp_output = decoder_output.transpose(0, 1)
        return bv_output, hfp_output

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
            self.bv_position.normal_()
            self.bv_start_token.normal_()
            self.hfp_start_token.normal_()
