import unittest
import abcgan  # import interface components directly from abcgan
import abcgan.constants as const
import numpy as np
import os
import h5py
from abcgan.interface import estimate_drivers, load_h5_data, generate_multi
from abcgan.interface import hellinger_scores_hfp, hellinger_scores_bv

dir_path = os.path.dirname(os.path.realpath(__file__))
fname = os.path.join(dir_path, "..", "tutorials", "tutorial_all.h5")


with h5py.File(fname, 'r') as hf:
    nSamples = hf['Drivers'][abcgan.driver_names[0]].shape[0]


def fake_drivers(n):
    return np.exp(np.random.normal(size=(n, const.n_driver)))


def fake_bvs(n):
    bvs = [np.random.uniform(low=const.bv_thresholds[i, 0],
                             high=const.bv_thresholds[i, 1],
                             size=(n, const.max_alt)) for i in range(const.n_bv)]
    return np.stack(bvs, axis=-1)


def fake_hfp(n):
    hfps = [np.random.uniform(low=const.hfp_thresholds[i, 0],
                              high=const.hfp_thresholds[i, 1],
                              size=n) for i in range(const.n_hfp)]
    return np.stack(hfps, axis=-1)[:, None, :]


def fake_lidar_bvs(n):
    return np.exp(np.random.normal(size=(n, const.max_alt_lidar, const.n_lidar_bv)))


class TestInterface(unittest.TestCase):

    def test_driver_estimation(self):
        drivers = fake_drivers(5)
        est_drs = estimate_drivers(drivers)
        self.assertEqual(est_drs.shape, drivers.shape)

    def test_generator(self):
        drivers = fake_drivers(5)
        bvs = abcgan.generate(drivers, verbose=0)
        self.assertEqual(bvs.shape[0], drivers.shape[0])

    def test_generator_zscale(self):
        drivers = fake_drivers(5)
        bvs = abcgan.generate(drivers, return_z_scale=True, verbose=0)
        self.assertEqual(bvs.shape[0], drivers.shape[0])
        self.assertTrue(np.isclose(np.std(bvs), 1, rtol=1, atol=1))
        self.assertTrue(np.isclose(np.mean(bvs), 0, rtol=1, atol=1))

    def test_generator_bv_inputs(self):
        top_alt = np.random.randint(const.max_alt - 1)
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)[:, :top_alt, :]
        G_bvs = abcgan.generate(drivers, measurements=bvs, verbose=0)
        self.assertEqual(G_bvs.shape[0], drivers.shape[0])
        self.assertTrue((bvs == G_bvs[:, :top_alt, :]).all())
        self.assertEqual(G_bvs.shape, (len(bvs), const.max_alt, const.n_bv))

    def test_hfp_generator(self):
        drivers = fake_drivers(5)
        bvs, hfps, b = abcgan.generate(drivers, generate_hfps=True, verbose=0)
        self.assertEqual(bvs.shape, (drivers.shape[0], const.max_alt, const.n_bv))
        self.assertEqual(hfps.shape, (drivers.shape[0], const.n_waves, const.n_hfp))
        self.assertEqual(b.shape, (drivers.shape[0], ))

    def test_generator_hfp_zscale(self):
        drivers = fake_drivers(5)
        bvs, hfps, b = abcgan.generate(drivers, return_z_scale=True, generate_hfps=True, verbose=0)
        self.assertTrue(np.isclose(np.std(bvs), 1, rtol=1, atol=1))
        self.assertTrue(np.isclose(np.mean(bvs), 0, rtol=1, atol=1))
        self.assertTrue(np.isclose(np.std(hfps), 1, rtol=1, atol=1))
        self.assertTrue(np.isclose(np.mean(hfps), 0, rtol=1, atol=1))

    def test_hfp_generator_bv_inputs(self):
        top_alt = np.random.randint(const.max_alt - 1)
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)[:, :top_alt, :]
        G_bvs, hfps, b = abcgan.generate(drivers, measurements=bvs,
                                         generate_hfps=True, verbose=0)
        self.assertEqual(G_bvs.shape, (drivers.shape[0], const.max_alt, const.n_bv))
        self.assertTrue((bvs == G_bvs[:, :top_alt, :]).all())
        self.assertEqual(hfps.shape, (drivers.shape[0], const.n_waves, const.n_hfp))
        self.assertEqual(b.shape, (drivers.shape[0], ))

    def test_bv_gen_multi_inputs(self):
        n_repeat = 5
        drivers = fake_drivers(5)
        G_bvs = generate_multi(drivers, n_repeat=n_repeat, verbose=0)
        self.assertEqual(G_bvs.shape, (drivers.shape[0], n_repeat, const.max_alt, const.n_bv))

    def test_gen_multi_bv_inputs(self):
        n_repeat = 5
        top_alt = np.random.randint(const.max_alt - 1)
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)[:, :top_alt, :]
        G_bvs = generate_multi(drivers, bvs=bvs, n_repeat=n_repeat, verbose=0)
        self.assertEqual(G_bvs.shape, (drivers.shape[0], n_repeat, const.max_alt, const.n_bv))
        self.assertTrue((bvs == G_bvs[:, 0, :top_alt, :]).all())

    def test_hfp_gen_multi_inputs(self):
        n_repeat = 5
        drivers = fake_drivers(5)
        G_bvs, G_hfps, G_bs = generate_multi(drivers, generate_hfps=True, n_repeat=n_repeat, verbose=0)
        self.assertEqual(G_bvs.shape, (drivers.shape[0], n_repeat, const.max_alt, const.n_bv))
        self.assertEqual(G_hfps.shape, (drivers.shape[0], n_repeat, const.n_waves, const.n_hfp))
        self.assertEqual(G_bs.shape, (drivers.shape[0], n_repeat, ))

    def test_hfp_gen_multi_bv_inputs(self):
        n_repeat = 5
        top_alt = np.random.randint(const.max_alt - 1)
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)[:, :top_alt, :]
        G_bvs, G_hfps, G_bs = generate_multi(drivers, bvs=bvs, n_repeat=n_repeat,
                                             generate_hfps=True, verbose=0)
        self.assertEqual(G_bvs.shape, (drivers.shape[0], n_repeat, const.max_alt, const.n_bv))
        self.assertTrue((bvs == G_bvs[:, 0, :top_alt, :]).all())
        self.assertEqual(G_hfps.shape, (drivers.shape[0], n_repeat, const.n_waves, const.n_hfp))
        self.assertEqual(G_bs.shape, (drivers.shape[0], n_repeat, ))

    def test_hfp_discriminator(self):
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)
        hfps = fake_hfp(5)
        bv_scores, hfp_scores = abcgan.discriminate(drivers, bvs, hfps=hfps)
        self.assertEqual(bv_scores.shape, (drivers.shape[0], bvs.shape[1]))
        self.assertEqual(hfp_scores.shape, (drivers.shape[0], hfps.shape[1]))

    def test_lidar_generator(self):
        drivers = fake_drivers(5)
        bvs = abcgan.generate(drivers, n_alt=const.max_alt_lidar,
                              bv_model='bv_lidar_gan', bv_type='lidar', verbose=0)
        self.assertEqual(bvs.shape[0], drivers.shape[0])

    def test_lidar_discriminator(self):
        drivers = fake_drivers(5)
        bvs = fake_lidar_bvs(5)
        disc = abcgan.discriminate(drivers, bvs, bv_model='bv_lidar_gan', bv_type='lidar')
        self.assertEqual(disc.shape, (drivers.shape[0], bvs.shape[1]))

    def test_discriminator(self):
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)
        disc = abcgan.discriminate(drivers, bvs, bv_model='bv_gan')
        self.assertEqual(disc.shape, (drivers.shape[0], bvs.shape[1]))

    def test_stack_drivers(self):
        with h5py.File(fname, 'r') as hf:
            drivers = abcgan.stack_drivers(hf['Drivers'])
        self.assertTrue(isinstance(drivers, np.ndarray))
        self.assertEqual(drivers.shape, (nSamples, const.n_driver))

    def test_stack_lidar_bvs(self):
        with h5py.File(fname, 'r') as hf:
            bvs = abcgan.stack_bvs(hf['BackgroundValues'], bv_type='lidar')
        self.assertTrue(isinstance(bvs, np.ndarray))
        self.assertEqual(bvs.shape,
                         (nSamples, const.max_alt_lidar, const.n_lidar_bv))

    def test_stack_bvs_radar(self):
        with h5py.File(fname, 'r') as hf:
            bvs = abcgan.stack_bvs(hf['BackgroundValues'], bv_type='radar')
        self.assertTrue(isinstance(bvs, np.ndarray))
        self.assertEqual(bvs.shape,
                         (nSamples, const.max_alt, const.n_bv))

    def test_load_h5_data_bv(self):
        drivers, bvs, alt_mask, unix_time = load_h5_data(fname)
        n_samples = len(drivers)
        self.assertTrue(len(drivers) == len(bvs) == len(alt_mask) == len(unix_time))
        self.assertEqual(drivers.shape, (n_samples, const.n_driver))
        self.assertEqual(bvs.shape, (n_samples, const.max_alt, const.n_bv))
        self.assertEqual(alt_mask.shape, (n_samples, const.max_alt))
        self.assertEqual(unix_time.shape, (n_samples,))
        n_samples = 500
        drivers, bvs, alt_mask, unix_time = load_h5_data(fname, n_samples=n_samples)
        self.assertTrue(len(drivers) == len(bvs) == len(alt_mask) == len(unix_time))
        self.assertEqual(drivers.shape, (n_samples, const.n_driver))
        self.assertEqual(bvs.shape, (n_samples, const.max_alt, const.n_bv))
        self.assertEqual(alt_mask.shape, (n_samples, const.max_alt))
        self.assertEqual(unix_time.shape, (n_samples,))
        drivers, bvs, alt_mask, hfps, wave_mask, unix_time = load_h5_data(fname, load_hfp=True)
        n_samples = len(drivers)
        self.assertTrue(len(drivers) == len(bvs) == len(alt_mask) == len(unix_time) == len(hfps) == len(wave_mask))
        self.assertEqual(drivers.shape, (n_samples, const.n_driver))
        self.assertEqual(bvs.shape, (n_samples, const.max_alt, const.n_bv))
        self.assertEqual(alt_mask.shape, (n_samples, const.max_alt))
        self.assertEqual(unix_time.shape, (n_samples,))
        n_samples = 500
        drivers, bvs, alt_mask, hfps, wave_mask, unix_time = load_h5_data(fname, n_samples=n_samples, load_hfp=True)
        self.assertTrue(len(drivers) == len(bvs) == len(alt_mask) == len(unix_time) == len(hfps) == len(wave_mask))
        self.assertEqual(drivers.shape, (n_samples, const.n_driver))
        self.assertEqual(bvs.shape, (n_samples, const.max_alt, const.n_bv))
        self.assertEqual(alt_mask.shape, (n_samples, const.max_alt))
        self.assertEqual(unix_time.shape, (n_samples,))

    def test_bv_hellinger_scoring(self):
        n_samples = 50
        drivers, bvs, alt_mask, unix_time = load_h5_data(fname, n_samples=n_samples)
        G_bvs = abcgan.generate(drivers, verbose=0)
        dist = hellinger_scores_bv(bvs, G_bvs, alt_mask)
        self.assertEqual(dist.shape, (const.max_alt, const.n_bv))
        dist = hellinger_scores_bv(bvs, G_bvs, alt_mask, filter_length=0)
        self.assertEqual(dist.shape, (const.max_alt, const.n_bv))
        dist = hellinger_scores_bv(bvs, G_bvs, alt_mask, z_scale=False)
        self.assertEqual(dist.shape, (const.max_alt, const.n_bv))

    def test_hfp_hellinger_scoring(self):
        n_samples = 50
        drivers, bvs, alt_mask, hfps, wave_mask, unix_time = load_h5_data(fname, n_samples=n_samples, load_hfp=True)
        G_bvs, G_hfps, G_bs = abcgan.generate(drivers, generate_hfps=True, verbose=0)
        Gb_mask = (G_bs < np.random.uniform(size=len(G_bs)))[:, None]
        dist = hellinger_scores_hfp(hfps, G_hfps, r_mask=wave_mask, f_mask=Gb_mask)
        self.assertEqual(dist.shape, (const.n_waves, const.n_hfp))
        dist = hellinger_scores_hfp(hfps, G_hfps, r_mask=wave_mask, f_mask=Gb_mask, filter_length=0)
        self.assertEqual(dist.shape, (const.n_waves, const.n_hfp))
        dist = hellinger_scores_hfp(hfps, G_hfps, r_mask=wave_mask, f_mask=Gb_mask, z_scale=False)
        self.assertEqual(dist.shape, (const.n_waves, const.n_hfp))

    def test_type(self):
        with h5py.File(fname, 'r') as hf:
            driver_dict = {k: v[()] for k, v in hf['Drivers'].items()}
            bv_dict = {k: v[()] for k, v in
                       hf['BackgroundValues'].items()}
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
                       hf['BackgroundValues'].items()}
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
                       hf['BackgroundValues'].items()}
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
