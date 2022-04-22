import unittest
import abcgan.constants as const
from abcgan import transforms as trans
import numpy as np


def fake_drivers(n):
    return np.exp(np.random.normal(size=(n, const.n_driver)))


def fake_bvs(n):
    return np.exp(np.random.normal(size=(n, const.max_alt, const.n_bv)))


def fake_hfp(n):
    hfp = np.zeros((n, const.n_waves, const.n_hfp))
    for i in range(const.n_hfp):
        for j in range(const.n_waves):
            hfp[:, j, i] = np.random.uniform(low=const.hfp_thresholds[i, 0],
                                             high=const.hfp_thresholds[i, 1],
                                             size=n)
    hfp[np.random.randint(low=0, high=n), :, :] = np.nan
    return hfp


class TestTransforms(unittest.TestCase):

    def test_scale_driver(self):
        drivers = fake_drivers(10)
        driver_feat = trans.scale_driver(drivers)
        self.assertEqual(driver_feat.shape,
                         (drivers.shape[0], const.n_driver_feat))

    def test_scale_bv(self):
        batch_size = 10
        bvs = fake_bvs(batch_size)
        bv_feat, valid_mask = trans.scale_bv(bvs)
        self.assertEqual(bv_feat.shape,
                         (bvs.shape[0], bvs.shape[1],
                          const.n_bv_feat))
        self.assertEqual(valid_mask.shape, (batch_size,))

    def test_scale_hfp(self):
        batch_size = 10
        hfp = fake_hfp(batch_size)
        hfp_feat, valid_mask = trans.scale_hfp(hfp)
        self.assertEqual(hfp_feat.shape,
                         (hfp.shape[0], const.n_waves, const.n_hfp))
        self.assertEqual(valid_mask.shape, (batch_size,))

    def test_get_driver(self):
        drivers = fake_drivers(10)
        driver_feat = trans.scale_driver(drivers)
        new_drivers = trans.get_driver(driver_feat)
        self.assertEqual(drivers.shape, new_drivers.shape)
        self.assertTrue(np.allclose(drivers, new_drivers))

    def test_get_driver_subset(self):
        driver_names = const.driver_names[4:8]
        drivers = fake_drivers(10)[:, :len(driver_names)]
        driver_feat = trans.scale_driver(drivers, driver_names=driver_names)
        new_drivers = trans.get_driver(driver_feat, driver_names=driver_names)
        self.assertEqual(drivers.shape, new_drivers.shape)
        self.assertTrue(np.allclose(drivers, new_drivers))

    def test_get_hfp(self):
        hfp = fake_hfp(10)
        hfp_feat, valid_mask = trans.scale_hfp(hfp)
        new_hfp = trans.get_hfp(hfp_feat)
        self.assertEqual(new_hfp.shape, hfp.shape)
        self.assertTrue(np.allclose(hfp, new_hfp, equal_nan=True))

    def test_get_bv(self):
        bvs = fake_bvs(10)
        bv_feat, valid_mask = trans.scale_bv(bvs)
        new_bvs = trans.get_bv(bv_feat)
        self.assertEqual(new_bvs.shape[1], const.max_alt)
        self.assertTrue(np.allclose(bvs, new_bvs))

    def test_padding(self):
        bvs = fake_bvs(10)
        bvs = bvs[:, :const.max_alt - 5, :]
        bv_feat, valid_mask = trans.scale_bv(bvs)
        self.assertEqual(bv_feat.shape[1], const.max_alt)

    def test_shrink(self):
        bvs = fake_bvs(10)
        bvs = np.concatenate((bvs, bvs), axis=1)
        bv_feat, valid_mask = trans.scale_bv(bvs)
        self.assertEqual(bv_feat.shape[1], const.max_alt)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
