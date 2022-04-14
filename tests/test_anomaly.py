import unittest
import os
from abcgan.interface import load_h5_data
from abcgan.anomaly import anomaly_score
from abcgan.interface import generate_multi
import abcgan.constants as const

dir_path = os.path.dirname(os.path.realpath(__file__))
fname = os.path.join(dir_path, "..", "tutorials", "tutorial_all.h5")


class TestAnomaly(unittest.TestCase):
    def test_alt_anomaly_score_bv(self):
        n_repeat = 5
        n_samples = 5
        drivers, bvs, _, _ = load_h5_data(fname, n_samples=n_samples)
        G_bvs = generate_multi(drivers, n_repeat=n_repeat, verbose=0)
        anomalies = anomaly_score(bvs, G_bvs, nd_est=True)
        self.assertEqual(anomalies.shape, (n_samples, const.max_alt))

    def test_anomaly_score_bv_1d(self):
        n_repeat = 5
        n_samples = 5
        drivers, bvs, _, _ = load_h5_data(fname, n_samples=n_samples)
        G_bvs = generate_multi(drivers, n_repeat=n_repeat)
        anomalies = anomaly_score(bvs, G_bvs)
        self.assertEqual(anomalies.shape, (n_samples, const.max_alt, const.n_bv))

    def test_anomaly_score_hfp(self):
        n_repeat = 5
        n_samples = 5
        drivers, _, _, hfps, _, _ = load_h5_data(fname, n_samples=n_samples, load_hfp=True)
        _, G_hfps, _ = generate_multi(drivers, n_repeat=n_repeat, generate_hfps=True)
        anomalies = anomaly_score(hfps, G_hfps)
        self.assertEqual(anomalies.shape, (drivers.shape[0], const.n_waves, const.n_hfp))

    def test_nd_anomaly_score_hfp(self):
        n_repeat = 5
        n_samples = 5
        drivers, _, _, hfps, _, _ = load_h5_data(fname, n_samples=n_samples, load_hfp=True)
        _, G_hfps, _ = generate_multi(drivers, n_repeat=n_repeat, generate_hfps=True)
        anomalies = anomaly_score(hfps, G_hfps, nd_est=True)
        self.assertEqual(anomalies.shape, (drivers.shape[0], const.n_waves))

    def test_anomaly_score_hfp_input_bvs(self):
        n_repeat = 5
        n_samples = 5
        drivers, bvs, _, hfps, _, _ = load_h5_data(fname, n_samples=n_samples, load_hfp=True)
        _, G_hfps, _ = generate_multi(drivers, bvs=bvs, n_repeat=n_repeat, generate_hfps=True)
        anomalies = anomaly_score(hfps, G_hfps)
        self.assertEqual(anomalies.shape, (drivers.shape[0], const.n_waves, const.n_hfp))


if __name__ == '__main__':
    unittest.main()
