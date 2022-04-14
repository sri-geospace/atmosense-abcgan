import unittest
from abcgan.mean_estimation import Transformer
from abcgan.bv_model import Generator, Critic, Driver_Generator, Driver_Critic
from abcgan.loader import get_fake_dataset
from torch.utils.data import DataLoader
import abcgan.constants as const
import torch


def get_fake_generator_data(bs=500, latent_dim=16):
    ds = get_fake_dataset(n_samples=bs)
    dl = DataLoader(ds, batch_size=bs, num_workers=0)
    dr, bv, mask = next(iter(dl))
    dr = dr.float()
    bv = bv.float()
    mask = ~mask
    noise = torch.randn(bs * const.max_alt, latent_dim).float()
    return dr, bv, mask, noise


def get_fake_critic_data(bs=500):
    ds = get_fake_dataset(n_samples=bs)
    dl = DataLoader(ds, batch_size=bs, num_workers=0)
    dr, bv, mask = next(iter(dl))
    bv = bv.float()
    return dr, bv, mask


class TestBVModel(unittest.TestCase):
    def test_generator(self):
        bs = 500
        gen_transformer = Transformer()
        gen_model = Generator(transformer=gen_transformer)
        dr, bv, mask, noise = get_fake_generator_data(bs, gen_model.latent_dim)
        fake_output = gen_model(dr, bv, ~mask, noise)
        self.assertEqual(fake_output.shape, bv.shape)

    def test_critic(self):
        bs = 500
        crit_transformer = Transformer()
        crit_model = Critic(transformer=crit_transformer)
        dr, fake_input, mask = get_fake_critic_data(bs)
        fake_pred = crit_model(fake_input, dr, fake_input, ~mask)
        self.assertEqual(fake_pred.shape, (fake_input.shape[0]*fake_input.shape[1], 1))

    def test_dr_generator(self):
        bs = 500
        gen_model = Driver_Generator()
        dr, _, _, _ = get_fake_generator_data(bs, gen_model.latent_dim)
        fake_output = gen_model(dr)
        self.assertEqual(fake_output.shape, dr.shape)

    def test_dr_critic(self):
        bs = 500
        crit_model = Driver_Critic()
        dr, _, _ = get_fake_critic_data(bs)
        fake_pred = crit_model(dr, dr)
        self.assertEqual(fake_pred.shape, (dr.shape[0], 1))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
