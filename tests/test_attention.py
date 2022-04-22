import unittest
import abcgan.constants as const
from abcgan.attention import collect_bv_attn_map, collect_hfp_attn_map
import numpy as np


def get_data(n):
    drivers = np.exp(np.random.normal(size=(n, const.n_driver)))
    bvs = np.exp(np.random.normal(size=(n, const.max_alt, const.n_bv)))
    return drivers, bvs


class TestAttention(unittest.TestCase):

    def test_collect_bv_attention(self):
        drivers, bvs = get_data(10)
        attn_map = collect_bv_attn_map(drivers, bvs)
        self.assertEqual(attn_map.shape, (len(drivers), const.max_alt, const.max_alt))

    def test_collect_hfp_attention(self):
        drivers, bvs = get_data(10)
        attn_map = collect_hfp_attn_map(drivers, bvs)
        self.assertEqual(attn_map.shape, (len(drivers), const.n_waves, const.max_alt))


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
