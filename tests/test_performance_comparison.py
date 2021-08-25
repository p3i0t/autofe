import unittest
import time
import torch
import jax.numpy as jnp
from jax import jit, random
import numpy as np

from autofe.functions.pytorch import _ts_day_sum as _ts_day_sum_p


# from autofe.functions.pytorch import _ts_min_sum as _ts_min_sum_p
from autofe.functions.jax import _ts_day_sum as _ts_day_sum_j
# from autofe.functions.jax import _ts_min_sum as _ts_min_sum_j

def f(a: int):
    print(f'wo ri {a}')

class PerformanceComparison(unittest.TestCase):
    def test_day_sum_comparison(self):
        n_runs = 5
        d = 5
        n_day = 200
        n_min = 20
        n_symbol = 500
        x = torch.randn(n_day, n_min, n_symbol)


        s = time.time()
        # for _ in range(n_runs):
        #     _ts_day_sum_p(x, d)
        print(time.time()-s)


        key = random.PRNGKey(0)
        x = random.normal(key, (n_day, n_min, n_symbol))

        print(x.shape)
        _ts_day_sum_jit = jit(_ts_day_sum_j)
        s = time.time()
        for _ in range(n_runs):
            _ts_day_sum_jit(x, d).block_until_ready()
        print(time.time()-s)




        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
    # for idx in range(10):
    #     timeit.timeit(stmt='f(idx)', number=1, globals=globals())
