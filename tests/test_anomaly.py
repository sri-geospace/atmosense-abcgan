import unittest
import os
import numpy as np
from abcgan.interface import load_h5_data, load_wtec_h5
from abcgan.anomaly import anomaly_score_bv, anomaly_score_hfp, anomaly_score_wtec
from abcgan.interface import generate_multi_bv, generate_multi_hfp, generate_multi_wtec
import abcgan.constants as const

dir_path = os.path.dirname(os.path.realpath(__file__))
fname = os.path.join(dir_path, "..", "tutorials", "tutorial_isr.h5")
wtec_fname = os.path.join(dir_path, "..", "tutorials", "wtec_tutorial.h5")
data_dir = os.path.join(dir_path, "..", "tutorials")


class TestAnomaly(unittest.TestCase):
    def test_joint_anomaly_score_bv(self):
        n_repeat = 5
        n_samples = 5
        drivers, bvs, _, _ = load_h5_data(fname, n_samples=n_samples)
        top_alt = np.random.randint(low=0, high=const.max_alt)
        G_bvs = generate_multi_bv(drivers, bvs=bvs[:, :top_alt, :], n_repeat=n_repeat, verbose=0)
        anomalies = anomaly_score_bv(bvs[:, top_alt: ], G_bvs[:, :, top_alt:], method='joint')
        self.assertEqual(anomalies.shape, (n_samples, const.max_alt - top_alt))

    def test_marginal_anomaly_score_bv(self):
        n_repeat = 5
        n_samples = 5
        drivers, bvs, _, _ = load_h5_data(fname, n_samples=n_samples)
        G_bvs = generate_multi_bv(drivers, n_repeat=n_repeat, verbose=0)
        anomalies = anomaly_score_bv(bvs, G_bvs, method='marginal')
        self.assertEqual(anomalies.shape, (n_samples, const.max_alt, const.n_bv))

    def test_joint_anomaly_score_hfp(self):
        n_repeat = 5
        n_samples = 5
        drivers, bvs, _, hfps, _, _ = load_h5_data(fname, n_samples=n_samples, load_hfp=True)
        G_hfps, _ = generate_multi_hfp(drivers, bvs=bvs, n_repeat=n_repeat, verbose=0)
        anomalies = anomaly_score_hfp(hfps, G_hfps, method='joint')
        self.assertEqual(anomalies.shape, (drivers.shape[0], const.n_waves))

    def test_marginal_anomaly_score_hfp(self):
        n_repeat = 5
        n_samples = 5
        drivers, bvs, _, hfps, _, _ = load_h5_data(fname, n_samples=n_samples, load_hfp=True)
        G_hfps, _ = generate_multi_hfp(drivers, bvs=bvs, n_repeat=n_repeat, verbose=0)
        anomalies = anomaly_score_hfp(hfps, G_hfps, method='marginal')
        self.assertEqual(anomalies.shape, (drivers.shape[0], const.n_waves, const.n_hfp))

    def test_joint_anomaly_score_hfp_cond_bv(self):
        n_repeat = 5
        n_samples = 5
        drivers, bvs, _, hfps, _, _ = load_h5_data(fname, n_samples=n_samples, load_hfp=True)
        top_alt = np.random.randint(low=1, high=const.max_alt)
        G_hfps, _ = generate_multi_hfp(drivers, bvs=bvs[:, :top_alt, :], n_repeat=n_repeat, verbose=0)
        anomalies = anomaly_score_hfp(hfps, G_hfps, method='joint')
        self.assertEqual(anomalies.shape, (drivers.shape[0], const.n_waves))

    def test_marginal_anomaly_score_hfp_cond_bv(self):
        n_repeat = 5
        n_samples = 5
        drivers, bvs, _, hfps, _, _ = load_h5_data(fname, n_samples=n_samples, load_hfp=True)
        top_alt = np.random.randint(low=1, high=const.max_alt)
        G_hfps, _ = generate_multi_hfp(drivers, bvs=bvs[:, :top_alt, :], n_repeat=n_repeat, verbose=0)
        anomalies = anomaly_score_hfp(hfps, G_hfps, method='marginal')
        self.assertEqual(anomalies.shape, (drivers.shape[0], const.n_waves, const.n_hfp))

    def test_joint_anomaly_score_wtec(self):
        n_repeat = 20
        n_samples = 100
        for tid_type in const.wtec_dict.keys():
            data = load_wtec_h5(fname=wtec_fname, locations=const.wtec_default_location,
                                tid_type=tid_type, n_samples=n_samples)
            for loc, res in data.items():
                drivers, wtec, unix_time = res["drivers"], res["wtecs"], res["utc"]
                G_wtec = generate_multi_wtec(drivers, tid_type=tid_type, location=loc, n_repeat=n_repeat, verbose=0)
                anomalies = anomaly_score_wtec(wtec, G_wtec, tid_type=tid_type, method='joint')
                self.assertTrue(np.isfinite(anomalies).all())
                self.assertEqual(anomalies.shape, (drivers.shape[0],))

    def test_marginal_anomaly_score_wtec(self):
        n_repeat = 20
        n_samples = 100
        for tid_type in const.wtec_dict.keys():
            data = load_wtec_h5(fname=wtec_fname, locations=const.wtec_default_location,
                                tid_type=tid_type, n_samples=n_samples)
            for loc, res in data.items():
                drivers, wtec, unix_time = res["drivers"], res["wtecs"], res["utc"]
                G_wtec = generate_multi_wtec(drivers, tid_type=tid_type, location=loc, n_repeat=n_repeat, verbose=0)
                anomalies = anomaly_score_wtec(wtec, G_wtec, tid_type=tid_type, method='marginal')
                self.assertTrue(np.isfinite(anomalies).all())
                self.assertEqual(anomalies.shape, (drivers.shape[0], const.n_wtec))


if __name__ == '__main__':
    unittest.main()
