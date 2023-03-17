import unittest
import abcgan  # import interface components directly from abcgan
import abcgan.constants as const
import numpy as np
import os
import h5py
import abcgan.transforms as trans
from abcgan.interface import estimate_drivers, load_h5_data, generate_multi_hfp, load_wtec_h5
from abcgan.interface import hellinger_scores_hfp, hellinger_scores_bv, hellinger_scores_wtec
from abcgan.interface import generate_wtec, generate_multi_bv, average_wtec
from abcgan.interface import generate_bvs, generate_hfps, generate_multi_wtec

dir_path = os.path.dirname(os.path.realpath(__file__))
fname = os.path.join(dir_path, "..", "tutorials", "tutorial_isr.h5")
wtec_fname = os.path.join(dir_path, "..", "tutorials", f"wtec_{const.wtec_default_dataset}.h5")
wtec_fnames = [os.path.join(dir_path, "..", "tutorials", f"wtec_{ds}.h5") for ds in const.wtec_datasets_names]


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


def fake_wtec(n, ds=const.wtec_default_dataset):
    wtec = np.zeros((n, const.n_wtec))
    for i in range(const.n_wtec):
        wtec[:, i] = np.random.uniform(low=const.wtec_zscale_dict[ds]['meas_ranges'][i, 0],
                                       high=const.wtec_zscale_dict[ds]['meas_ranges'][i, 0],
                                       size=n)
    return wtec


class TestInterface(unittest.TestCase):

    def test_driver_estimation(self):
        drivers = fake_drivers(5)
        est_drs = estimate_drivers(drivers)
        self.assertEqual(est_drs.shape, drivers.shape)

    def test_generator(self):
        drivers = fake_drivers(5)
        bvs = generate_bvs(drivers, verbose=0)
        self.assertEqual(bvs.shape, (len(drivers), const.max_alt, const.n_bv))

    def test_generator_bv_inputs(self):
        top_alt = np.random.randint(low=0, high=const.max_alt)
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)[:, :top_alt, :]
        G_bvs = generate_bvs(drivers, bv_measurements=bvs, verbose=0)
        self.assertEqual(G_bvs.shape[0], drivers.shape[0])
        self.assertTrue((bvs == G_bvs[:, :top_alt, :]).all())
        self.assertEqual(G_bvs.shape, (len(bvs), const.max_alt, const.n_bv))

    def test_hfp_generator(self):
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)
        hfps, b = generate_hfps(drivers, bvs, verbose=0)
        self.assertEqual(hfps.shape, (drivers.shape[0], const.n_waves, const.n_hfp))
        self.assertEqual(b.shape, (drivers.shape[0], ))

    def test_wtec_generator(self):
        drivers = fake_drivers(5)
        wtecs = generate_wtec(drivers, verbose=0)
        self.assertEqual(wtecs.shape, (drivers.shape[0], const.n_wtec))
        self.assertTrue(not (np.isnan(wtecs).any()))
        wtec_feats = generate_wtec(drivers, return_z_scale=True, verbose=0)
        self.assertEqual(wtec_feats.shape, (drivers.shape[0], const.n_wtec))
        self.assertTrue(not (np.isnan(wtec_feats).any()))

    def test_bv_generator_missing_alts(self):
        top_alt = np.random.randint(low=1, high=const.max_alt)
        bvs = fake_bvs(5)[:, :top_alt, :]
        drivers = fake_drivers(5)
        G_bvs = generate_bvs(drivers, bv_measurements=bvs, verbose=0)
        self.assertEqual(G_bvs.shape, (drivers.shape[0], const.max_alt, const.n_bv))
        self.assertTrue(~(np.isnan(G_bvs).any()))
        self.assertTrue((bvs == G_bvs[:, :top_alt, :]).all())

    def test_hfp_generator_missing_alts(self):
        top_alt = np.random.randint(low=1, high=const.max_alt)
        bvs = fake_bvs(5)[:, :top_alt, :]
        tmp_bvs = bvs.copy()
        drivers = fake_drivers(5)
        G_hfps, b = generate_hfps(drivers, bv_measurements=tmp_bvs, verbose=0)
        self.assertTrue((bvs == tmp_bvs).all())
        self.assertTrue(~(np.isnan(G_hfps).any()))
        self.assertEqual(G_hfps.shape, (drivers.shape[0], const.n_waves, const.n_hfp))
        self.assertEqual(b.shape, (drivers.shape[0], ))

    def test_bv_gen_multi_inputs(self):
        n_repeat = 5
        drivers = fake_drivers(5)
        G_bvs = generate_multi_bv(drivers, n_repeat=n_repeat, verbose=0)
        self.assertEqual(G_bvs.shape, (drivers.shape[0], n_repeat, const.max_alt, const.n_bv))
        self.assertTrue(~(np.isnan(G_bvs).any()))

    def test_gen_multi_missing_alts(self):
        n_repeat = 5
        top_alt = np.random.randint(low=1, high=const.max_alt)
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)[:, :top_alt, :]
        G_bvs = generate_multi_bv(drivers, bvs=bvs, n_repeat=n_repeat, verbose=0)
        self.assertEqual(G_bvs.shape, (drivers.shape[0], n_repeat, const.max_alt, const.n_bv))
        self.assertTrue((bvs == G_bvs[:, 0, :top_alt, :]).all())
        self.assertTrue(~(np.isnan(G_bvs).any()))

    def test_hfp_gen_multi(self):
        n_repeat = 5
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)
        G_hfps, G_bs = generate_multi_hfp(drivers, bvs=bvs, n_repeat=n_repeat, verbose=0)
        self.assertEqual(G_hfps.shape, (drivers.shape[0], n_repeat, const.n_waves, const.n_hfp))
        self.assertEqual(G_bs.shape, (drivers.shape[0], n_repeat, ))
        self.assertTrue(~(np.isnan(G_hfps).any()))
        self.assertTrue(~(np.isnan(G_bs).any()))

    def test_hfp_gen_multi_missing_alts(self):
        n_repeat = 5
        top_alt = np.random.randint(low=1, high=const.max_alt)
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)[:, :top_alt, :]
        G_hfps, G_bs = generate_multi_hfp(drivers, bvs=bvs, n_repeat=n_repeat, verbose=0)
        self.assertEqual(G_hfps.shape, (drivers.shape[0], n_repeat, const.n_waves, const.n_hfp))
        self.assertEqual(G_bs.shape, (drivers.shape[0], n_repeat, ))
        self.assertTrue(~(np.isnan(G_hfps).any()))
        self.assertTrue(~(np.isnan(G_bs).any()))

    def test_wtec_gen_multi(self):
        n_repeat = 5
        drivers = fake_drivers(5)
        G_wtec = generate_multi_wtec(drivers, n_repeat=n_repeat, verbose=0)
        self.assertEqual(G_wtec.shape, (drivers.shape[0], n_repeat, const.n_wtec))

    def test_hfp_discriminator(self):
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)
        hfps = fake_hfp(5)
        bv_scores, hfp_scores = abcgan.discriminate(drivers, bvs, hfps=hfps)
        self.assertEqual(bv_scores.shape, (drivers.shape[0], bvs.shape[1]))
        self.assertEqual(hfp_scores.shape, (drivers.shape[0], hfps.shape[1]))

    def test_lidar_generator(self):
        drivers = fake_drivers(5)
        bvs = generate_bvs(drivers, n_alt=const.max_alt_lidar,
                           bv_model='bv_lidar_gan', bv_type='lidar',
                           verbose=0)
        self.assertEqual(bvs.shape, (len(drivers), const.max_alt_lidar, const.n_lidar_bv))

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

    def test_load_tec_h5(self):
        for filename, ds in zip(wtec_fnames, const.wtec_datasets_names):
            drivers, wtec, unix_time = load_wtec_h5(filename, dataset_name=ds)
            n_samples = len(drivers)
            self.assertTrue(len(drivers) == len(wtec) == len(unix_time))
            self.assertEqual(drivers.shape, (n_samples, const.n_wtec_dr))
            self.assertEqual(wtec.shape, (n_samples, const.n_wtec))
            self.assertEqual(unix_time.shape, (n_samples,))
            n_samples = 500
            drivers, wtec, unix_time = load_wtec_h5(filename, n_samples=n_samples, dataset_name=ds)
            self.assertTrue(len(drivers) == len(wtec) == len(unix_time))
            self.assertEqual(drivers.shape, (n_samples, const.n_wtec_dr))
            self.assertEqual(wtec.shape, (n_samples, const.n_wtec))
            self.assertEqual(unix_time.shape, (n_samples,))

    def test_bv_hellinger_scoring(self):
        n_samples = 50
        drivers, bvs, alt_mask, unix_time = load_h5_data(fname, n_samples=n_samples)
        top_alt = np.random.randint(low=1, high=const.max_alt)
        G_bvs = generate_bvs(drivers, bvs[:, :top_alt, :], verbose=0)
        dist = hellinger_scores_bv(bvs, G_bvs, alt_mask)
        self.assertEqual(dist.shape, (const.max_alt, const.n_bv))
        self.assertTrue(~(np.isnan(dist).any()))
        G_bvs = generate_bvs(drivers, verbose=0)
        dist = hellinger_scores_bv(bvs, G_bvs, alt_mask, filter_length=0)
        self.assertEqual(dist.shape, (const.max_alt, const.n_bv))
        self.assertTrue(~(np.isnan(dist).any()))
        dist = hellinger_scores_bv(bvs, G_bvs, alt_mask, z_scale=False)
        self.assertEqual(dist.shape, (const.max_alt, const.n_bv))
        self.assertTrue(~(np.isnan(dist).any()))

    def test_hfp_hellinger_scoring(self):
        n_samples = 50
        drivers, bvs, alt_mask, hfps, wave_mask, unix_time = load_h5_data(fname, n_samples=n_samples, load_hfp=True)
        top_alt = np.random.randint(low=1, high=const.max_alt)
        G_hfps, G_bs = generate_hfps(drivers, bvs[:, :top_alt, :], verbose=0)
        Gb_mask = (G_bs < np.random.uniform(size=len(G_bs)))[:, None]
        dist = hellinger_scores_hfp(hfps, G_hfps, r_mask=wave_mask, f_mask=Gb_mask)
        self.assertTrue(~(np.isnan(dist).any()))
        self.assertEqual(dist.shape, (const.n_waves, const.n_hfp))
        dist = hellinger_scores_hfp(hfps, G_hfps, r_mask=wave_mask, f_mask=Gb_mask, filter_length=0)
        self.assertTrue(~(np.isnan(dist).any()))
        self.assertEqual(dist.shape, (const.n_waves, const.n_hfp))
        dist = hellinger_scores_hfp(hfps, G_hfps, r_mask=wave_mask, f_mask=Gb_mask, z_scale=False)
        self.assertTrue(~(np.isnan(dist).any()))
        self.assertEqual(dist.shape, (const.n_waves, const.n_hfp))

    def test_wtec_hellinger_scoring(self):
        n_samples = 50
        for filename, ds in zip(wtec_fnames, const.wtec_datasets_names):
            drivers, wtec, unix_time = load_wtec_h5(wtec_fname, dataset_name=ds, n_samples=n_samples)
            wtec_fake = fake_wtec(n_samples, ds=ds)
            dist = hellinger_scores_wtec(wtec, wtec_fake, dataset_name=ds)
            self.assertEqual(dist.shape, (const.n_wtec,))
            dist = hellinger_scores_wtec(wtec, wtec_fake, dataset_name=ds, filter_length=0)
            self.assertEqual(dist.shape, (const.n_wtec,))
            dist = hellinger_scores_wtec(wtec, wtec_fake, dataset_name=ds, filter_length=2, z_scale=False)
            self.assertEqual(dist.shape, (const.n_wtec,))

    def test_wtec_averaging(self):
        n_samples = 50
        drivers, wtec, unix_time = load_wtec_h5(wtec_fname, n_samples=n_samples)
        wtec_avg = average_wtec(wtec)
        self.assertEqual(wtec_avg.shape, wtec.shape)
        self.assertTrue(np.isclose((np.nanmean(wtec) - np.nanmean(wtec_avg))/np.nanmean(wtec), 0, atol=1e-2))
        self.assertTrue(np.isclose((np.nanstd(wtec) - np.nanstd(wtec_avg))/np.nanstd(wtec), 0, atol=1e-2))

    def test_zscale_wtec_averaging(self):
        n_samples = 50
        drivers, wtec, unix_time = load_wtec_h5(wtec_fname, n_samples=n_samples)
        wtec_feat = trans.scale_wtec(wtec)[0]
        wtec_feat_avg = average_wtec(wtec_feat, z_scale_input=True)
        self.assertEqual(wtec_feat_avg.shape, wtec_feat.shape)
        self.assertTrue(np.isclose((np.nanmean(wtec_feat) - np.nanmean(wtec_feat_avg)), 0, atol=1e-1))
        self.assertTrue(np.isclose((np.nanstd(wtec_feat) - np.nanstd(wtec_feat_avg)), 0, atol=1e-1))

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
