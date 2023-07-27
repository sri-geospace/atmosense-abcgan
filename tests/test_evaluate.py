import unittest
import numpy as np
import os

from .test_interface import fake_wtec, fake_hfp, fake_bvs
import abcgan.constants as const
from abcgan.interface import generate_bvs, generate_hfps, load_h5_data, load_wtec_h5
from abcgan.evaluate import hellinger_scores_hfp, hellinger_scores_bv,\
    hellinger_scores_wtec, conditional_wtec_scores, get_clusters

dir_path = os.path.dirname(os.path.realpath(__file__))
fname = os.path.join(dir_path, "..", "tutorials", "tutorial_isr.h5")
wtec_fname = os.path.join(dir_path, "..", "tutorials", "wtec_tutorial.h5")


class TestEvaluate(unittest.TestCase):

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
        for tid_type in const.wtec_dict.keys():
            locations = [d[7:] for d in const.wtec_dataset_names if d.find(tid_type) == 0
                         and d[7:] in const.wtec_sites.keys()]
            data = load_wtec_h5(fname=wtec_fname, locations=locations, tid_type=tid_type, n_samples=n_samples, start_utc='random')
            for loc, res in data.items():
                drivers, wtec, unix_time = res["drivers"], res["wtecs"], res["utc"]
                wtec_fake = fake_wtec(n_samples, tid_type=tid_type)
                dist = hellinger_scores_wtec(wtec, wtec_fake, tid_type=tid_type)
                self.assertEqual(dist.shape, (const.n_wtec,))
                dist = hellinger_scores_wtec(wtec, wtec_fake, tid_type=tid_type, filter_length=0)
                self.assertEqual(dist.shape, (const.n_wtec,))
                dist = hellinger_scores_wtec(wtec, wtec_fake, tid_type=tid_type, use_kde=True)
                self.assertEqual(dist.shape, (const.n_wtec,))
                dist = hellinger_scores_wtec(wtec, wtec_fake, tid_type=tid_type, filter_length=2, z_scale=False)
                self.assertEqual(dist.shape, (const.n_wtec,))

    def test_wtec_conditional_scoring(self):
        n_samples = 100
        n_clusters = 3
        tid_type = const.wtec_default_tid_type
        locations = [d[7:] for d in const.wtec_dataset_names if d.find(tid_type) == 0
                     and d[7:] in const.wtec_sites.keys()]
        data = load_wtec_h5(fname=wtec_fname, locations=locations, tid_type=tid_type, n_samples=n_samples, start_utc='random')
        for loc, res in data.items():
            drivers, wtec, unix_time = res["drivers"], res["wtecs"], res["utc"]
            wtec_fake = fake_wtec(n_samples, tid_type=tid_type)
            dist = conditional_wtec_scores(drivers, wtec, wtec_fake, n_clusters=n_clusters, tid_type=tid_type)
            self.assertEqual(dist.shape, (drivers.shape[-1], n_clusters, const.n_wtec,))
            dist = conditional_wtec_scores(drivers, wtec, wtec_fake, tid_type=tid_type, n_clusters=n_clusters, filter_length=0)
            self.assertEqual(dist.shape, (drivers.shape[-1], n_clusters, const.n_wtec,))
            dist = conditional_wtec_scores(drivers, wtec, wtec_fake, tid_type=tid_type, n_clusters=n_clusters, use_kde=True)
            self.assertEqual(dist.shape, (drivers.shape[-1], n_clusters, const.n_wtec,))
            dist = conditional_wtec_scores(drivers, wtec, wtec_fake, n_clusters=n_clusters,
                                           tid_type=tid_type, filter_length=2, z_scale=True)
            self.assertEqual(dist.shape, (drivers.shape[-1], n_clusters, const.n_wtec,))

    def test_get_clusters(self):
        n_samples = 100
        n_clusters = np.arange(1, 5)
        tid_type = const.wtec_default_tid_type
        locations = const.wtec_default_location
        data = load_wtec_h5(fname=wtec_fname, locations=locations, tid_type=tid_type, n_samples=n_samples, start_utc='random')
        for loc, res in data.items():
            drivers, wtec, unix_time = res["drivers"], res["wtecs"], res["utc"]
            for c in n_clusters:
                clusters = get_clusters(drivers, c)
                self.assertTrue((np.unique(clusters) == np.arange(c)).all())


if __name__ == '__main__':
    unittest.main()
