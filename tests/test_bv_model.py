import unittest
from abcgan.mean_estimation import Transformer
from abcgan.bv_model import Generator, Critic, Driver_Generator, Driver_Critic
from torch.utils.data import DataLoader, TensorDataset
import abcgan.constants as const
import abcgan.transforms as trans
import numpy as np
import torch


def fake_drivers(n):
    return np.exp(np.random.normal(size=(n, const.n_driver)))


def fake_bvs(n):
    bvs = [np.random.uniform(low=const.bv_thresholds[i, 0],
                             high=const.bv_thresholds[i, 1],
                             size=(n, const.max_alt)) for i in range(const.n_bv)]
    return np.stack(bvs, axis=-1)


def fake_alt_mask(n):
    alt_mask = np.ones((n, const.max_alt), dtype=bool)
    end_alt = np.random.randint(low=2, high=const.max_alt, size=(n,))
    for i in range(n):
        alt_mask[i, end_alt[i]:] = False
    return alt_mask


def get_fake_bv_data(bs=500, latent_dim=16):
    drivers = torch.tensor(trans.scale_driver(fake_drivers(2 * bs)), dtype=torch.float)
    bvs = torch.tensor(trans.scale_bv(fake_bvs(2 * bs))[0], dtype=torch.float)
    alt_mask = torch.tensor(fake_alt_mask(2 * bs), dtype=torch.bool)
    ds = TensorDataset(drivers, bvs, alt_mask)
    dl = DataLoader(ds, batch_size=bs, num_workers=0)
    dr, bv, mask = next(iter(dl))
    dr = dr.float()
    bv = bv.float()
    mask = ~mask
    noise = torch.randn(bs * const.max_alt, latent_dim).float()
    return dr, bv, mask, noise


class TestBVModel(unittest.TestCase):
    def test_generator(self):
        bs = 500
        gen_transformer = Transformer()
        gen_model = Generator(transformer=gen_transformer)
        dr, bv, mask, noise = get_fake_bv_data(bs, gen_model.latent_dim)
        fake_output = gen_model(dr, bv, ~mask, noise)
        self.assertEqual(fake_output.shape, bv.shape)

    def test_critic(self):
        bs = 500
        crit_transformer = Transformer()
        crit_model = Critic(transformer=crit_transformer)
        dr, fake_input, mask, _ = get_fake_bv_data(bs)
        fake_pred = crit_model(fake_input, dr, fake_input, ~mask)
        self.assertEqual(fake_pred.shape, (fake_input.shape[0]*fake_input.shape[1], 1))

    def test_dr_generator(self):
        bs = 500
        gen_model = Driver_Generator()
        dr, _, _, _ = get_fake_bv_data(bs, gen_model.latent_dim)
        fake_output = gen_model(dr)
        self.assertEqual(fake_output.shape, dr.shape)

    def test_dr_critic(self):
        bs = 500
        crit_model = Driver_Critic()
        dr, _, _, _ = get_fake_bv_data(bs)
        fake_pred = crit_model(dr, dr)
        self.assertEqual(fake_pred.shape, (dr.shape[0], 1))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
