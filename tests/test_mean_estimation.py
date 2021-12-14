import unittest
from abcgan.mean_estimation import Transformer
import abcgan.constants as const
import torch
import numpy as np
from torch import Tensor


def fake_drivers(n):
    driver = torch.rand((n, const.n_driver_feat))
    return driver


def fake_bvs(n_alt, n):
    bvs = torch.rand((n, n_alt, const.n_bv_feat))
    return bvs


def fake_src_key_mask(n_alt, n):
    mask = torch.ones(n_alt, n)
    null_batch = np.random.randint(0, n - 1, 2)
    null_alt = np.random.randint(0, n_alt - 1, 2)
    mask[:, null_batch] = 0
    mask[null_alt, :] = 0
    mask = (mask == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    mask = torch.gt(mask, 0)
    return mask


class TestTransformerEstimation(unittest.TestCase):

    def test_transformer(self):
        n_alt = const.max_alt
        model = Transformer(n_alt=n_alt)

        driver = fake_drivers(10)
        bvs = fake_bvs(n_alt, 10)
        est_bv = model(driver, bvs)
        self.assertEqual(est_bv.shape,
                         (bvs.shape[0], bvs.shape[1],
                          model.d_model))

    def test_scr_key_mask(self):
        n = 10
        n_alt = const.max_alt
        model = Transformer(n_alt=n_alt)

        mask = fake_src_key_mask(n_alt, n)
        driver = fake_drivers(n)
        bvs = fake_bvs(n_alt, n)
        est_bv = model(driver, bvs, mask)
        self.assertEqual(est_bv.shape,
                         (bvs.shape[0], bvs.shape[1],
                          model.d_model))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
