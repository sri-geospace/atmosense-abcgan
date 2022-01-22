import unittest
import abcgan.constants as const
from abcgan import transforms as trans
import numpy as np


def fake_drivers(n):
    return np.exp(np.random.normal(size=(n, const.n_driver)))


def fake_bvs(n):
    return np.exp(np.random.normal(size=(n, const.max_alt, const.n_bv)))


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
