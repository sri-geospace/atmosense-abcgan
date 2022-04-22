import abcgan.constants as const
import torch.nn as nn
import torch
from torch.nn.init import xavier_uniform_


class Generator(nn.Module):
    """
    Generator Class

    Parameters
    -----------
    transformer: torch.nn.Module
        transformer for the generator
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
                 img_dim=const.n_bv_feat,
                 hidden_dim=128):
        super(Generator, self).__init__()

        cond_dim = transformer.d_out + latent_dim
        self.transformer = transformer

        self.input_args = {
            'n_layers': n_layers,
            'latent_dim': latent_dim,
            'img_dim': img_dim,
            'hidden_dim': hidden_dim,
        }
        # Build the neural network
        layers = [nn.Linear(cond_dim, hidden_dim),
                  nn.ReLU(inplace=True)]
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, img_dim))
        self.gen_model = nn.Sequential(*layers)

        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.img_dim = img_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self._reset_parameters()

    def forward(self, driver_src, bv_src, src_key_mask=None, noise=None):
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
        src_key_mask: torch.Tensor, optional
            mask for bv features from data loader (n_alt, n_batch)
        noise: torch.Tensor, optional
            a noise tensor with dimensions (n_batch, latent_dim)
        """
        if src_key_mask is None:
            src_key_mask = torch.zeros(
                bv_src.shape[0], bv_src.shape[1],
                dtype=torch.bool, device=bv_src.device)
        est_bv = self.transformer(driver_src, bv_src, src_key_mask)
        est_bv = est_bv.flatten(0, 1)
        if noise is None:
            noise = torch.randn(*est_bv.shape[:-1], self.latent_dim,
                                dtype=driver_src.dtype,
                                device=driver_src.device)
        fake_output = self.gen_model(torch.cat((est_bv, noise), 1))
        fake_output = fake_output.reshape(bv_src.shape)
        return fake_output

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)


class Critic(nn.Module):
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
                 img_dim=const.n_bv_feat,
                 hidden_dim=128):
        super(Critic, self).__init__()

        cond_dim = transformer.d_out + img_dim
        self.transformer = transformer

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
        self._reset_parameters()

    def forward(self, bv_features, driver_src, real, src_key_mask=None):
        """
        Function for completing a forward pass of the critic: Given an image
        tensor, returns a 1-dimension tensor representing a fake/real
        prediction.

        Parameters
        ---------------
        bv_features: torch.Tensor
            a flattened image tensor with dimension
            (n_batch, max_alt, n_bv_feat)
        driver_src: torch.Tensor
            tensor of driver features from data loader (n_batch, n_dr_feat)
        real: torch.Tensor
            tensor of bv features from data loader (n_batch, n_alt, n_bv_feat)
        src_key_mask: torch.Tensor, optional
            mask for bv features from data loader (n_batch, n_alt)
        """
        if src_key_mask is None:
            src_key_mask = torch.zeros(
                bv_features.shape[0], bv_features.shape[1],
                dtype=torch.bool, device=bv_features.device)
        curr_bv = bv_features.flatten(0, 1)
        est_real = self.transformer(driver_src, real, src_key_mask)
        cond_feat = torch.cat((est_real.flatten(0, 1), curr_bv), 1)
        pred = self.crit_model(cond_feat)
        return pred

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)


class Driver_Generator(nn.Module):
    """
    Generator Class

    Parameters
    -----------
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
                 n_layers=2,
                 latent_dim=16,
                 img_dim=const.n_driver_feat,
                 hidden_dim=64):
        super(Driver_Generator, self).__init__()

        cond_dim = img_dim + latent_dim
        self.input_args = {
            'n_layers': n_layers,
            'latent_dim': latent_dim,
            'img_dim': img_dim,
            'hidden_dim': hidden_dim,
        }
        # Build the neural network
        layers = [nn.Linear(cond_dim, hidden_dim),
                  nn.ReLU(inplace=True)]
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, img_dim))
        self.gen_model = nn.Sequential(*layers)

        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.img_dim = img_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self._reset_parameters()

    def forward(self, dr_prev, noise=None):
        """
        forward pass of the generator for driver augmentation:
        Given driver sample from the past and noise tensor,
        returns generated driver sample.
        Parameters
        --------------
        dr_prev: torch.Tensor
            tensor of past driver features from data loader (n_batch, n_dr_feat)
        noise: torch.Tensor, optional
            a noise tensor with dimensions (n_batch, latent_dim)
        """
        if noise is None:
            noise = torch.randn(dr_prev.shape[0], self.latent_dim,
                                dtype=dr_prev.dtype,
                                device=dr_prev.device)
        fake_output = self.gen_model(torch.cat((dr_prev, noise), 1))
        return fake_output

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)


class Driver_Critic(nn.Module):
    """
    Critic Class

    Parameters
    -------------
    n_layers: int
        number of layers in MLP
    img_dim: int
        the dimension of the images, fitted for the dataset used, a scalar
    hidden_dim: int
        the inner dimension, a scalar
    """

    def __init__(self,
                 n_layers=2,
                 img_dim=const.n_driver_feat,
                 hidden_dim=64):
        super(Driver_Critic, self).__init__()

        cond_dim = img_dim + img_dim
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
        self._reset_parameters()

    def forward(self, dr_src, dr_prev):
        """
        forward pass of the critic for driver augmentation: Given an image
        tensor, returns a 1-dimension tensor representing a fake/real
        prediction.

        Parameters
        ---------------
        dr_src: torch.Tensor
            tensor of driver features (n_batch, n_dr_feat)
        dr_prev: torch.Tensor
            tensor of past driver features  (n_batch, n_dr_feat)
        """
        cond_feat = torch.cat((dr_prev, dr_src), 1)
        pred = self.crit_model(cond_feat)
        return pred

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

