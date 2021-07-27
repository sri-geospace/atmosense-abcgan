import unittest
import abcgan  # import interface components directly from abcgan
import abcgan.constants as const
import numpy as np
import os
import h5py

dir_path = os.path.dirname(os.path.realpath(__file__))
fname = os.path.join(dir_path, "..", "tutorials", "tutorial.h5")

with h5py.File(fname, 'r') as hf:
    nSamples = hf['Drivers'][abcgan.driver_names[0]].shape[0]
    nAltitude = hf['BackgroundVariables'][abcgan.bv_names[0]].shape[1]


def fake_drivers(n):
    return np.exp(np.random.normal(size=(n, const.n_driver)))


def fake_bvs(n):
    return np.exp(np.random.normal(size=(n, const.max_alt, const.n_bv)))


class TestInterface(unittest.TestCase):

    def test_generator(self):
        drivers = fake_drivers(10)
        bvs = abcgan.generate(drivers)
        self.assertEqual(bvs.shape[0], drivers.shape[0])

    def test_discriminator(self):
        drivers = fake_drivers(10)
        bvs = fake_bvs(10)
        disc = abcgan.discriminate(drivers, bvs)
        self.assertEqual(disc.shape, (drivers.shape[0], bvs.shape[1]))

    def test_stack_drivers(self):
        with h5py.File(fname, 'r') as hf:
            drivers = abcgan.stack_drivers(hf['Drivers'])
        self.assertTrue(isinstance(drivers, np.ndarray))
        self.assertEqual(drivers.shape, (nSamples, const.n_driver))

    def test_stack_bvs(self):
        with h5py.File(fname, 'r') as hf:
            bvs = abcgan.stack_bvs(hf['BackgroundVariables'])
        self.assertTrue(isinstance(bvs, np.ndarray))
        self.assertEqual(bvs.shape,
                         (nSamples, nAltitude, const.n_bv))

    def test_type(self):
        with h5py.File(fname, 'r') as hf:
            driver_dict = {k: v[()] for k, v in hf['Drivers'].items()}
            bv_dict = {k: v[()] for k, v in
                       hf['BackgroundVariables'].items()}
            driver_dict[const.driver_names[0]] = None
            with self.assertRaises(ValueError):
                abcgan.stack_drivers(driver_dict)
            bv_dict[const.bv_names[0]] = None
            with self.assertRaises(ValueError):
                abcgan.stack_bvs(bv_dict)

    def test_missing_name(self):
        with h5py.File(fname, 'r') as hf:
            driver_dict = {k: v[()] for k, v in hf['Drivers'].items()}
            bv_dict = {k: v[()] for k, v in
                       hf['BackgroundVariables'].items()}
            del driver_dict[const.driver_names[0]]
            with self.assertRaises(KeyError):
                abcgan.stack_drivers(driver_dict)
            del bv_dict[const.bv_names[0]]
            with self.assertRaises(KeyError):
                abcgan.stack_bvs(bv_dict)

    def test_wrong_shape(self):
        with h5py.File(fname, 'r') as hf:
            driver_dict = {k: v[()] for k, v in hf['Drivers'].items()}
            bv_dict = {k: v[()] for k, v in
                       hf['BackgroundVariables'].items()}
            driver_dict[const.driver_names[0]] = \
                driver_dict[const.driver_names[0]][:10]
            with self.assertRaises(ValueError):
                abcgan.stack_drivers(driver_dict)
            bv_dict[const.bv_names[0]] = \
                bv_dict[const.bv_names[0]][:5]
            with self.assertRaises(ValueError):
                abcgan.stack_bvs(bv_dict)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
