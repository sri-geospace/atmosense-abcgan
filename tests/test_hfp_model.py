import unittest
from abcgan.hfp_model import HFP_Transformer, HFP_Critic, HFP_Generator
import torch
import abcgan.constants as const
from torch.utils.data import DataLoader, TensorDataset


def get_fake_data(bs=500, latent_dim=16):
    drivers = torch.randn((bs, const.n_driver_feat))
    bvs = torch.randn((bs, const.max_alt, const.n_bv_feat))
    src_mask = torch.ones((bs, const.max_alt), dtype=torch.bool)
    hfps = torch.randn((bs, const.n_waves, const.n_hfp_feat))
    tgt_mask = (torch.randint(10, size=(bs,)) > 4)[:, None]
    hfps[tgt_mask] = 0.0
    ds = TensorDataset(drivers, bvs, src_mask, hfps, tgt_mask)
    dl = DataLoader(ds, batch_size=bs, num_workers=0)
    dr, bv, smask, h, tmask = next(iter(dl))
    dr = dr.float()
    bv = bv.float()
    h = h.float()
    noise = torch.randn(h.shape[0] * h.shape[1],
                        latent_dim,
                        dtype=h.dtype)
    return dr, bv, smask, h, tmask, noise


class TestHFPGAN(unittest.TestCase):
    def test_hfp_generator(self):
        bs = 500
        gen_transformer = HFP_Transformer(output_b=True)
        gen_model = HFP_Generator(transformer=gen_transformer)
        dr, bv, src_mask, hfp, tgt_mask, noise = get_fake_data(bs, gen_model.latent_dim)
        fake_output, b = gen_model(dr, bv, hfp, ~src_mask, ~tgt_mask, noise)
        self.assertEqual(fake_output.shape, hfp.shape)
        self.assertEqual(b.shape[0], hfp.shape[0])
        self.assertTrue(torch.isnan(fake_output).sum() == 0)
        self.assertTrue(torch.isnan(b).sum() == 0)

    def test_critic(self):
        bs = 500
        crit_transformer = HFP_Transformer()
        crit_model = HFP_Critic(transformer=crit_transformer)
        dr, bv, src_mask, hfp, tgt_mask, noise = get_fake_data(bs, 16)
        fake_pred = crit_model(dr, bv, hfp, hfp, ~src_mask, ~tgt_mask)
        self.assertEqual(fake_pred.shape, (hfp.shape[0]*hfp.shape[1], 1))
        self.assertTrue(torch.isnan(fake_pred).sum() == 0)


if __name__ == '__main__':
    unittest.main()
