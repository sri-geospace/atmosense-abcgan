import unittest
import abcgan.constants as const
import numpy as np
import os
import itertools
import h5py
import abcgan.transforms as trans
from abcgan.interface import estimate_drivers, load_h5_data, generate_multi_hfp, load_wtec_h5
from abcgan.interface import generate_wtec, generate_multi_bv, average_wtec, stack_bvs, stack_drivers
from abcgan.interface import generate_bvs, generate_hfps, generate_multi_wtec, discriminate

dir_path = os.path.dirname(os.path.realpath(__file__))
fname = os.path.join(dir_path, "..", "tutorials", "tutorial_isr.h5")
wtec_fname = os.path.join(dir_path, "..", "tutorials", f"wtec_tutorial.h5")

with h5py.File(fname, 'r') as hf:
    nSamples = hf['Drivers'][const.driver_names[0]].shape[0]


def fake_drivers(n, n_drivers=const.n_driver):
    return np.exp(np.random.normal(size=(n, n_drivers)))


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


def fake_wtec(n, tid_type=const.wtec_default_tid_type):
    wtec = np.zeros((n, const.n_wtec))
    for i in range(wtec.shape[-1]):
        wtec[:, i] = np.random.uniform(low=const.wtec_dict[tid_type]['meas_ranges'][i, 0],
                                       high=const.wtec_dict[tid_type]['meas_ranges'][i, 1],
                                       size=n)
    return wtec


def get_combinations(tid_type, n_tests):
    combinations = []
    for i in range(1, len(const.wtec_sites.keys())):
        combinations.extend(list(itertools.combinations([loc for loc in list(const.wtec_sites.keys())
                                                         if f'{tid_type}_{loc}' in const.wtec_dataset_names],
                                                        i)))
    combinations = [list(c) for c in combinations]
    if len(combinations) > n_tests:
        rand_indexes = list(np.random.choice(np.arange(len(combinations)), replace=False, size=(n_tests,)).astype(int))
        combinations = [c for i, c in enumerate(combinations) if i in rand_indexes]
    return combinations


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
        for ds in const.wtec_dataset_names:
            tid_type, loc = ds[:6], ds[7:]
            drivers = fake_drivers(5, const.n_wtec_dr)
            wtecs = generate_wtec(drivers, location=loc, tid_type=tid_type, verbose=0)
            self.assertEqual(wtecs.shape, (drivers.shape[0], const.n_wtec))
            self.assertTrue(not (np.isnan(wtecs).any()))
            wtec_feats = generate_wtec(drivers, location=loc, tid_type=tid_type,
                                       return_z_scale=True, verbose=0)
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
        drivers = fake_drivers(5, const.n_wtec_dr)
        for ds in const.wtec_dataset_names:
            tid_type, loc = ds[:6], ds[7:]
            G_wtec = generate_multi_wtec(drivers, tid_type=tid_type, location=loc,
                                         n_repeat=n_repeat, verbose=0)
            self.assertEqual(G_wtec.shape, (drivers.shape[0], n_repeat, const.n_wtec))

    def test_hfp_discriminator(self):
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)
        hfps = fake_hfp(5)
        bv_scores, hfp_scores = discriminate(drivers, bvs, hfps=hfps)
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
        disc = discriminate(drivers, bvs, bv_model='bv_lidar_gan', bv_type='lidar')
        self.assertEqual(disc.shape, (drivers.shape[0], bvs.shape[1]))

    def test_discriminator(self):
        drivers = fake_drivers(5)
        bvs = fake_bvs(5)
        disc = discriminate(drivers, bvs, bv_model='bv_gan')
        self.assertEqual(disc.shape, (drivers.shape[0], bvs.shape[1]))

    def test_stack_drivers(self):
        with h5py.File(fname, 'r') as hf:
            drivers = stack_drivers(hf['Drivers'])
        self.assertTrue(isinstance(drivers, np.ndarray))
        self.assertEqual(drivers.shape, (nSamples, const.n_driver))

    def test_stack_lidar_bvs(self):
        with h5py.File(fname, 'r') as hf:
            bvs = stack_bvs(hf['BackgroundValues'], bv_type='lidar')
        self.assertTrue(isinstance(bvs, np.ndarray))
        self.assertEqual(bvs.shape,
                         (nSamples, const.max_alt_lidar, const.n_lidar_bv))

    def test_stack_bvs_radar(self):
        with h5py.File(fname, 'r') as hf:
            bvs = stack_bvs(hf['BackgroundValues'], bv_type='radar')
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

    def test_load_tec_h5_start_random(self):
        combinations = get_combinations(const.wtec_default_tid_type, n_tests=2)
        n_samples = [None, np.random.randint(low=1, high=5000)]
        for locs, nb in zip(combinations, n_samples):
            data = load_wtec_h5(fname=wtec_fname, locations=locs, tid_type=const.wtec_default_tid_type,
                                start_utc='random', n_samples=nb)
            self.assertTrue(list(data.keys()) == locs)
            for loc, res in data.items():
                drivers, wtec, unix_time = res["drivers"], res["wtecs"], res["utc"]
                self.assertTrue(len(drivers) == len(wtec) == len(unix_time))
                self.assertEqual(drivers.shape[-1], const.n_wtec_dr)
                self.assertEqual(wtec.shape[-1], const.n_wtec)

    def test_load_tec_h5(self):
        n_test = 2
        for tid_type in list(const.wtec_dict.keys()):
            combinations = get_combinations(tid_type, n_test)
            for locs in combinations:
                data = load_wtec_h5(fname=wtec_fname, locations=locs, tid_type=tid_type)
                self.assertTrue(list(data.keys()) == locs)
                for loc, res in data.items():
                    drivers, wtec, unix_time = res["drivers"], res["wtecs"], res["utc"]
                    n_samples = len(drivers)
                    self.assertTrue(len(drivers) == len(wtec) == len(unix_time))
                    self.assertEqual(drivers.shape, (n_samples, const.n_wtec_dr))
                    self.assertEqual(wtec.shape, (n_samples, const.n_wtec))
                    self.assertEqual(unix_time.shape, (n_samples,))
        n_samples = 500
        locations = [d[7:] for d in const.wtec_dataset_names if d.find(const.wtec_default_tid_type) == 0
                     and d[7:] in const.wtec_sites.keys()]
        data = load_wtec_h5(fname=wtec_fname, locations=locations,
                            tid_type=const.wtec_default_tid_type,
                            n_samples=n_samples)
        for loc, res in data.items():
            drivers, wtec, unix_time = res["drivers"], res["wtecs"], res["utc"]
            self.assertTrue(len(drivers) == len(wtec) == len(unix_time))
            self.assertEqual(drivers.shape, (n_samples, const.n_wtec_dr))
            self.assertEqual(wtec.shape, (n_samples, const.n_wtec))
            self.assertEqual(unix_time.shape, (n_samples,))

    def test_load_tec_h5_context(self):
        n_test = 2
        n_context = 1
        for tid_type in list(const.wtec_dict.keys()):
            combinations = get_combinations(tid_type, n_test)
            for locs in combinations:
                data = load_wtec_h5(fname=wtec_fname, locations=locs, tid_type=tid_type, n_context=n_context)
                self.assertTrue(list(data.keys()) == locs)
                for loc, res in data.items():
                    drivers, wtec, unix_time, ctx_wtecs = res["drivers"], res["wtecs"], res["utc"], res["ctx_wtecs"]
                    n_samples = len(drivers)
                    self.assertTrue(len(drivers) == len(wtec) == len(unix_time) == len(ctx_wtecs))
                    self.assertEqual(drivers.shape, (n_samples, const.n_wtec_dr))
                    self.assertEqual(wtec.shape, (n_samples, const.n_wtec))
                    self.assertEqual(ctx_wtecs.shape, (n_samples, n_context, const.n_wtec))
                    self.assertEqual(unix_time.shape, (n_samples,))

    def test_wtec_averaging(self):
        n_samples = 50
        data = load_wtec_h5(fname=wtec_fname, locations=const.wtec_default_location,
                            tid_type=const.wtec_default_tid_type,
                            n_samples=n_samples)
        for loc, res in data.items():
            drivers, wtec, unix_time = res["drivers"], res["wtecs"], res["utc"]
            wtec_avg = average_wtec(wtec, tid_type=const.wtec_default_tid_type)
            self.assertEqual(wtec_avg.shape, wtec.shape)
            self.assertTrue(np.isclose((np.nanmean(wtec) - np.nanmean(wtec_avg))/np.nanmean(wtec), 0, atol=1e-2))
            self.assertTrue(np.isclose((np.nanstd(wtec) - np.nanstd(wtec_avg))/np.nanstd(wtec), 0, atol=1e-2))

    def test_zscale_wtec_averaging(self):
        n_samples = 50
        data = load_wtec_h5(fname=wtec_fname, locations=const.wtec_default_location,
                            tid_type=const.wtec_default_tid_type,
                            n_samples=n_samples)
        for loc, res in data.items():
            drivers, wtec, unix_time = res["drivers"], res["wtecs"], res["utc"]
            wtec_feat = trans.scale_wtec(wtec, tid_type=const.wtec_default_tid_type)[0]
            wtec_feat_avg = average_wtec(wtec_feat, tid_type=const.wtec_default_tid_type, z_scale_input=True)
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
                stack_drivers(driver_dict)
            bv_dict[const.bv_names[0]] = None
            with self.assertRaises(ValueError):
                stack_bvs(bv_dict)

    def test_missing_name(self):
        with h5py.File(fname, 'r') as hf:
            driver_dict = {k: v[()] for k, v in hf['Drivers'].items()}
            bv_dict = {k: v[()] for k, v in
                       hf['BackgroundValues'].items()}
            del driver_dict[const.driver_names[0]]
            with self.assertRaises(KeyError):
                stack_drivers(driver_dict)
            del bv_dict[const.bv_names[0]]
            with self.assertRaises(KeyError):
                stack_bvs(bv_dict)

    def test_wrong_shape(self):
        with h5py.File(fname, 'r') as hf:
            driver_dict = {k: v[()] for k, v in hf['Drivers'].items()}
            bv_dict = {k: v[()] for k, v in
                       hf['BackgroundValues'].items()}
            driver_dict[const.driver_names[0]] = \
                driver_dict[const.driver_names[0]][:10]
            with self.assertRaises(ValueError):
                stack_drivers(driver_dict)
            bv_dict[const.bv_names[0]] = \
                bv_dict[const.bv_names[0]][:5]
            with self.assertRaises(ValueError):
                stack_bvs(bv_dict)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
