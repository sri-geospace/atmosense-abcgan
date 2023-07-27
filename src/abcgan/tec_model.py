import abcgan.constants as const
from torch.nn.init import xavier_uniform_, xavier_normal_
import torch.nn as nn
import torch


class WTEC_Generator(nn.Module):
    """
    Generator Class

    Parameters
    -----------
    n_layers: int
        number of MLP layers
    latent_dim: int
        the dimension of the input latent vector
    feat_dim: int
        the dimension of the output features
    hidden_dim: int
        the inner dimension, a scalar
    """

    def __init__(self,
                 n_layers: int = 2,
                 latent_dim: int = 16,
                 feat_dim: int = const.n_wtec,
                 cond_dim: int = const.n_wtec_dr_feat,
                 hidden_dim: int = 64):
        super(WTEC_Generator, self).__init__()

        input_dim = cond_dim + latent_dim
        self.input_args = {
            'n_layers': n_layers,
            'latent_dim': latent_dim,
            'cond_dim': cond_dim,
            'feat_dim': feat_dim,
            'hidden_dim': hidden_dim,
        }
        # Build the neural network
        layers = [nn.Linear(input_dim, hidden_dim),
                  nn.ReLU(inplace=True)]
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, feat_dim))
        self.gen_model = nn.Sequential(*layers)

        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.feat_dim = feat_dim
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self._reset_parameters()

    def forward(self, driver, noise=None):
        """
        forward pass of the generator for generating TEC waves:
        Given input drivers and noise tensor,
        returns generated TEC wave parameters for each time sample.
        Parameters
        --------------
        driver: torch.Tensor
            tensor of  driver features from data loader (n_batch, n_wtec_dr_feat)
        noise: torch.Tensor, optional
            a noise tensor with dimensions (n_batch, latent_dim)

        Returns
        --------------
        fake_output: torch.Tensor
            generated tec waves (n_batch x n_wtec_feats)
        """
        if noise is None:
            noise = torch.randn(driver.shape[0],
                                self.latent_dim,
                                dtype=driver.dtype,
                                device=driver.device)
        fake_output = self.gen_model(torch.cat((driver, noise), 1))
        return fake_output

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)


class WTEC_Critic(nn.Module):
    """
    Critic Class

    Parameters
    -------------
    n_layers: int
        number of layers in MLP
    feat_dim: int
        the dimension of wtec features fitted for the dataset used, a scalar
    hidden_dim: int
        the inner dimension, a scalar
    """

    def __init__(self,
                 n_layers: int = 2,
                 feat_dim: int = const.n_wtec,
                 cond_dim: int = const.n_wtec_dr_feat,
                 hidden_dim: int = 64):
        super(WTEC_Critic, self).__init__()

        input_dim = cond_dim + 1 * feat_dim
        self.input_args = {
            'n_layers': n_layers,
            'feat_dim': feat_dim,
            'cond_dim': cond_dim,
            'hidden_dim': hidden_dim,
        }
        # Build the neural network
        layers = [nn.Linear(input_dim, hidden_dim),
                  nn.ReLU(inplace=True)]
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, 1))
        self.crit_model = nn.Sequential(*layers)

        self.n_layers = n_layers
        self.feat_dim = feat_dim
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self._reset_parameters()

    def forward(self, drivers, wtec_feats):
        """
        forward pass of the critic for tec wave given an image
        tensor, returns a 1-dimension tensor representing a fake/real
        prediction.

        Parameters
        ---------------
        drivers: torch.Tensor
            tensor of driver features (n_batch, n_wtec_dr_feat)
        wtec_feats: torch.Tensor
            tensor of fake or real tec feats (n_batch, n_wtec_waves, n_wtec_feat)
        """

        cond_feat = torch.cat((drivers, wtec_feats), 1)
        pred = self.crit_model(cond_feat)
        return pred

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
