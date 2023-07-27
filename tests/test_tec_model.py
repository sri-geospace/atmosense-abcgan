import unittest
from abcgan.tec_model import WTEC_Generator, WTEC_Critic
import abcgan.transforms as trans
import abcgan.constants as const
import torch
import numpy as np


def fake_wtec_data(n, tid_type):
    dr_feats = torch.randn(n, const.n_wtec_dr_feat, dtype=torch.float)
    wtec_real = np.zeros((n, const.n_wtec))
    wtec_fake = np.zeros((n, const.n_wtec))
    for i in range(const.n_wtec):
        wtec_real[:, i] = np.random.uniform(low=const.wtec_dict[tid_type]['meas_ranges'][i, 0],
                                            high=const.wtec_dict[tid_type]['meas_ranges'][i, 1],
                                            size=n)
        wtec_fake[:, i] = np.random.uniform(low=const.wtec_dict[tid_type]['meas_ranges'][i, 0],
                                            high=const.wtec_dict[tid_type]['meas_ranges'][i, 1],
                                            size=n)
    wtec_real = trans.scale_wtec(wtec_real, tid_type=tid_type)[0]
    wtec_real = torch.tensor(wtec_real, dtype=torch.float)
    wtec_fake = trans.scale_wtec(wtec_fake, tid_type=tid_type)[0]
    wtec_fake = torch.tensor(wtec_fake, dtype=torch.float)
    return dr_feats, wtec_real, wtec_fake


class TestTECModel(unittest.TestCase):
    def test_wtec_generator(self):
        bs = 10
        gen_model = WTEC_Generator()
        dr, wtec, _ = fake_wtec_data(bs, const.wtec_default_tid_type)
        gen_wtec = gen_model(dr)
        self.assertEqual(gen_wtec.shape, wtec.shape)

    def test_wtec_critic(self):
        bs = 10
        crit_model = WTEC_Critic()
        dr, real_wtec, fake_wtec = fake_wtec_data(bs, const.wtec_default_tid_type)
        fake_pred = crit_model(dr, fake_wtec)
        self.assertEqual(fake_pred.shape, (fake_wtec.shape[0], 1))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
