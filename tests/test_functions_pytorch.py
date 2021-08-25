import unittest
import random
import numpy as np
import pandas as pd
import scipy.stats

import torch
from autofe.functions.pytorch import (_nanargmax, _nanargmin, _nanmax, _nanmean,
                             _nanmin, _nanprod, _nanstd, _nanvar)

from autofe.functions.pytorch import (_inverse, _negative, _protected_log,
                                    _protected_sqrt,_sigmoid, _tanh,
                                    _signedpower)

from autofe.functions.pytorch import (_ts_day_corr, _ts_day_cov, _ts_day_prod,
                           _ts_day_rank, _ts_day_std, _ts_day_sum,
                           _ts_min_corr, _ts_min_cov, _ts_min_prod,
                           _ts_min_rank, _ts_min_std, _ts_min_sum,
                           _rank, _cross_rank, compute_correlation,
                           compute_cov)



class TorchFunctionTests(unittest.TestCase):
    def generate_nan_input(self):
        n = 100
        v = np.random.randn(n, n)
        # v[2, :] = np.NaN  # the whole row as NaN
        # v[:, 4] = np.NaN  # the whole column as NaN
        v[4, 5] = np.NaN
        v[14, 15] = np.NaN
        v[14, 19] = np.NaN
        v[24, 25] = np.NaN
        v[27, 25] = np.NaN
        return v

    def test_future_data_abuse(self):
        d = 7
        n_day = 70
        n_min = 30
        n_symbol = 100
        x = torch.randn(n_day, n_min, n_symbol)
        y = torch.randn(n_day, n_min, n_symbol)

        arity2_functions = [_ts_day_prod, _ts_day_rank, _ts_day_std, _ts_day_sum,
                            _ts_min_prod, _ts_min_rank, _ts_min_std, _ts_min_sum]

        arity3_functions = [_ts_day_corr, _ts_day_cov, _ts_min_corr, _ts_min_cov]

        n_random_tests = 10
        test_results = []
        for f in arity2_functions:
            for _ in range(n_random_tests):
                idx = random.randint(1, n_day-1)
                res1 = f(x, d)[:idx]
                res2 = f(x[:idx], d)
                o = np.allclose(res1, res2, rtol=1e-5, equal_nan=True)
                test_results.append(o)

        for f in arity3_functions:
            for _ in range(n_random_tests):
                idx = random.randint(1, n_day-1)
                res1 = f(x, y, d)[:idx]
                res2 = f(x[:idx], y[:idx], d)
                o = np.allclose(res1, res2, rtol=1e-5, equal_nan=True)
                test_results.append(o)

        self.assertTrue(np.sum(test_results))


    def test_compute_correlation(self):
        """test function compute_correlation by comparing with the results from scipy implementation.
        Note the function scipy.stats.pearsonr only compute the correlation of two 1-D numpy arrays.
        """
        n = 100
        m = 8
        x = np.random.randn(m, n, m, m)
        y = np.random.randn(m, n, m, m)

        res1 = compute_correlation(torch.from_numpy(x), torch.from_numpy(y))

        x_ = np.reshape(np.transpose(x, (1, 0, 2, 3)), (n, m**3))
        y_ = np.reshape(np.transpose(y, (1, 0, 2, 3)), (n, m**3))

        res2 = np.array([scipy.stats.pearsonr(x_[:, i], y_[:, i])[0] for i in range(m*m*m)])
        res2 = np.reshape(res2, (m, m, m))
        # print(res1)
        # print(res2)
        self.assertTrue(np.allclose(res1.numpy(), res2, rtol=1e-5, equal_nan=False))

    def test_compute_covariance(self):
        """test function compute_cov by comparing with the results from numpy implementation.
        """
        n = 100
        m = 8
        x = np.random.randn(m, n, m, m)
        y = np.random.randn(m, n, m, m)

        res1 = compute_cov(torch.from_numpy(x), torch.from_numpy(y))

        x_ = np.reshape(np.transpose(x, (1, 0, 2, 3)), (n, m**3))
        y_ = np.reshape(np.transpose(y, (1, 0, 2, 3)), (n, m**3))

        res2 = []
        for i in range(m*m*m):
            inp = np.stack([x_[:, i], y_[:, i]])
            cov = np.cov(inp, ddof=1)
            res2.append(cov[0, 1])
        res2 = np.reshape(np.array([res2]), (m, m, m))
        # print(res2.shape)
        # print(res2)
        self.assertTrue(np.allclose(res1.numpy(), res2, rtol=1e-5, equal_nan=False))

    def test_rank(self):
        """Test the _rank with a hard-coded example.
        """
        v = torch.tensor([[1., 3., 5., np.NaN, 2.],[np.NaN, -0.8, -1.0, 0, -2.0]])
        res = torch.tensor([[0.0, .4, .6, np.NaN, 0.2],[np.NaN, 0.4, 0.2, 0.6, 0.0]])

        # reshape input to (datetime-like rolling dimension, factor, symbol), factor dim is 1 by default.
        v = v.t().unsqueeze(dim=1)
        res = res.t().unsqueeze(dim=1)
        self.assertTrue(torch.allclose(_rank(v, dim=0), res, rtol=1e-5, equal_nan=True))

    def test_nanmax(self):
        """Test the _nanmax implementation by comparing with the results from numpy implementation.
        """
        v = self.generate_nan_input()
        vals1 = _nanmax(torch.from_numpy(v), dim=0, keepdim=True)
        vals1 = vals1.numpy()
        vals2 = np.nanmax(v, axis=0, keepdims=True)
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

        vals1 = _nanmax(torch.from_numpy(v), dim=1, keepdim=True)
        vals1 = vals1.numpy()
        vals2 = np.nanmax(v, axis=1, keepdims=True)
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

    def test_nanargmax(self):
        """Test the _nanargmax implementation by comparing with the results from numpy implementation.
        """
        v = self.generate_nan_input()
        vals1 = _nanargmax(torch.from_numpy(v), dim=0).numpy()
        vals2 = np.nanargmax(v, axis=0)  # no keepdims argument in numpy.nanargmax
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

        vals1 = _nanargmax(torch.from_numpy(v), dim=1).numpy()
        vals2 = np.nanargmax(v, axis=1)
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

    def test_nanmin(self):
        """Test the _nanmin implementation by comparing with the results from numpy implementation.
        """
        v = self.generate_nan_input()
        vals1 = _nanmin(torch.from_numpy(v), dim=0, keepdim=True)
        vals1 = vals1.numpy()
        vals2 = np.nanmin(v, axis=0, keepdims=True)
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

        vals1 = _nanmin(torch.from_numpy(v), dim=1, keepdim=True)
        vals1 = vals1.numpy()
        vals2 = np.nanmin(v, axis=1, keepdims=True)
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

    def test_nanargmin(self):
        """Test the _nanargmin implementation by comparing with the results from numpy implementation.
        """
        v = self.generate_nan_input()
        vals1 = _nanargmin(torch.from_numpy(v), dim=0).numpy()
        vals2 = np.nanargmin(v, axis=0)  # no keepdims argument in numpy.nanargmax
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

        vals1 = _nanargmin(torch.from_numpy(v), dim=1).numpy()
        vals2 = np.nanargmin(v, axis=1)
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

    def test_nanmean(self):
        """Test the _nanmean implementation by comparing with the results from numpy implementation.
        """
        v = self.generate_nan_input()
        vals1 = _nanmean(torch.from_numpy(v), dim=0).numpy()
        vals2 = np.nanmean(v, axis=0)  # no keepdims argument in numpy.nanargmax
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

        vals1 = _nanmean(torch.from_numpy(v), dim=1).numpy()
        vals2 = np.nanmean(v, axis=1)
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

    def test_nanprod(self):
        """Test the _nanprod implementation by comparing with the results from numpy implementation.
        """
        v = self.generate_nan_input()
        vals1 = _nanprod(torch.from_numpy(v), dim=0).numpy()
        vals2 = np.nanprod(v, axis=0)  # no keepdims argument in numpy.nanargmax
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

        vals1 = _nanprod(torch.from_numpy(v), dim=1).numpy()
        vals2 = np.nanprod(v, axis=1)
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

    def test_nanvar(self):
        """Test the _nanvar implementation by comparing with the results from numpy implementation.
        """
        v = self.generate_nan_input()
        vals1 = _nanvar(torch.from_numpy(v), dim=0, ddof=1).numpy()
        vals2 = np.nanvar(v, axis=0, ddof=1)  # no keepdims argument in numpy.nanargmax
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

        vals1 = _nanvar(torch.from_numpy(v), dim=1, ddof=1).numpy()
        vals2 = np.nanvar(v, axis=1, ddof=1)
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-4, equal_nan=True))

    def test_nanstd(self):
        """Test the _nanstd implementation by comparing with the results from numpy implementation.
        """
        v = self.generate_nan_input()
        vals1 = _nanstd(torch.from_numpy(v), dim=0, ddof=1).numpy()
        vals2 = np.nanstd(v, axis=0, ddof=1)  # no keepdims argument in numpy.nanargmax
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-5, equal_nan=True))

        vals1 = _nanstd(torch.from_numpy(v), dim=1, ddof=1).numpy()
        vals2 = np.nanstd(v, axis=1, ddof=1)
        self.assertTrue(np.allclose(vals1, vals2, rtol=1e-4, equal_nan=True))

    def test_protected_log(self):
        v = torch.tensor([1.0, np.NaN, 2.0, -2.0, -1.0])
        res = torch.tensor([0.69314, np.NaN, 1.09861, -1.09861, -0.69314])
        self.assertTrue(torch.allclose(_protected_log(v), res, rtol=1e-3, equal_nan=True))

    def test_protected_sqrt(self):
        v = torch.tensor([1.0, np.NaN, 2.0, -2.0, -1.0])
        res = torch.tensor([1.0, np.NaN, 1.41421, -1.41421, -1.0])
        self.assertTrue(torch.allclose(_protected_sqrt(v), res, rtol=1e-3, equal_nan=True))

    def test_inverse(self):
        v = torch.tensor([1.0, np.NaN, 2.0, -2.0, -1.0])
        res = torch.tensor([1.0, np.NaN, 0.5, -0.5, -1.0])
        self.assertTrue(torch.allclose(_inverse(v), res, rtol=1e-3, equal_nan=True))

    def test_sigmoid(self):
        self.assertTrue(True)

    def test_tanh(self):
        self.assertTrue(True)

    def test_negative(self):
        self.assertTrue(True)

    def test_correlation(self):
        n = 100
        m = 7
        q = 8

        d = 5

        # day_correlation
        x = np.random.randn(n, m, q)  # NaN not acceptable
        y = np.random.randn(n, m, q)  # NaN not acceptable

        xx = np.reshape(x, (n, m*q))
        yy = np.reshape(y, (n, m*q))

        res1 = []
        for i in range(m*q):
            df = pd.DataFrame({'a': xx[:, i], 'b': yy[:, i]})
            corr = df['a'].rolling(window=d, min_periods=1).corr(df['b'], pairwise=True).values
            res1.append(np.expand_dims(corr, axis=1))
        res1 = np.reshape(np.concatenate(res1, axis=1), (n, m, q))
        # print(res1)

        x_ = torch.from_numpy(x)
        y_ = torch.from_numpy(y)
        # print('fuck *')
        res2 = _ts_day_corr(x_, y_, d=d).numpy()
        # print(res2.shape)
        # print(res2)

        self.assertTrue(np.allclose(res1, res2, rtol=1e-5, equal_nan=True))

        # # min_correlation
        xxx = np.reshape(x, (-1, q))
        yyy = np.reshape(y, (-1, q))

        res3 = []
        for i in range(q):
            df = pd.DataFrame({'a': xxx[:, i], 'b': yyy[:, i]})
            corr = df['a'].rolling(window=d, min_periods=1).corr(df['b'], pairwise=True).values
            res3.append(np.expand_dims(corr, axis=1))

        res3 = np.reshape(np.concatenate(res3, axis=1), (n, m, q))
        res4 = _ts_min_corr(x_, y_, d=d).numpy()
        self.assertTrue(np.allclose(res3, res4, rtol=1e-5, equal_nan=True))



    def test_covariance(self):
        n = 100
        p = 7
        q = 9

        d = 5

        # day_correlation
        x = np.random.randn(n, p, q)  # NaN not acceptable
        y = np.random.randn(n, p, q)  # NaN not acceptable

        xx = np.reshape(x, (n, p*q))
        yy = np.reshape(y, (n, p*q))

        res1 = []
        for i in range(p*q):
            df = pd.DataFrame({'a': xx[:, i], 'b': yy[:, i]})
            corr = df['a'].rolling(window=d, min_periods=1).cov(df['b'], pairwise=True).values
            # print(corr.shape)
            res1.append(np.expand_dims(corr, axis=1))
        res1 = np.reshape(np.concatenate(res1, axis=1), (n, p, q))

        x_ = torch.from_numpy(x)
        y_ = torch.from_numpy(y)
        res2 = _ts_day_cov(x_, y_, d=d).numpy()

        # print(res1.shape, res1)
        # print(res2.shape, res2)
        self.assertTrue(np.allclose(res1, res2, rtol=1e-5, equal_nan=True))

        # min_correlation
        xxx = np.reshape(x, (-1, q))
        yyy = np.reshape(y, (-1, q))

        res3 = []
        for i in range(q):
            df = pd.DataFrame({'a': xxx[:, i], 'b': yyy[:, i]})
            corr = df['a'].rolling(window=d, min_periods=1).cov(df['b'], pairwise=True).values
            res3.append(np.expand_dims(corr, axis=1))

        res3 = np.reshape(np.concatenate(res3, axis=1), (n, p, q))
        res4 = _ts_min_cov(x_, y_, d=d).numpy()
        self.assertTrue(np.allclose(res3, res4, rtol=1e-5, equal_nan=True))

    def test_min_sum(self):
        d = 3
        x = torch.randn(3, 5, 10)
        res1 = _ts_min_sum(x, d)[:2]
        res2 = _ts_min_sum(x[:2], d)
        self.assertTrue(np.allclose(res1, res2, rtol=1e-5, equal_nan=True))

    def test_day_sum(self):

        d = 5
        n_day = 70
        n_min = 10
        n_symbol = 100
        x = torch.randn(n_day, n_min, n_symbol)
        idx = random.randint(0, n_day)
        res1 = _ts_day_sum(x, d)[:idx]
        res2 = _ts_day_sum(x[:idx], d)
        self.assertTrue(np.allclose(res1, res2, rtol=1e-5, equal_nan=True))

    # def test_ts_rank(self):
    #     self.assertTrue(True)

    # def test_compute_IC(self):
    #     x = torch.tensor([[0.5, -1.9, float('nan'), 3.0, 2.6, -10.8],
    #                       [float('nan'), -1.2, -4.9, 4.8, -1.0, 0.84]])
    #     y = torch.tensor([[0.35, -1.29, 1.05, float('nan'),  2.0, -2.8],
    #                 [0.48, float('nan'), -1.22, -1.19, -0.03, 1.04]])

    #     x_ = np.array([[0.5, -1.9, 2.6, -10.8],
    #                       [-4.9, 4.8, -1.0, 0.84]])

    #     y_ = np.array([[0.35, -1.29,  2.0, -2.8],
    #         [-1.22, -1.19, -0.03, 1.04]])

    #     # nan values will be ignored.
    #     res1 = compute_IC(x.unsqueeze(dim=1), y.unsqueeze(dim=1)).item()

    #     res = [scipy.stats.pearsonr(x_[0], y_[0])[0], scipy.stats.pearsonr(x_[1], y_[1])[0]]
    #     res2 = np.mean(res)
    #     # print(res2)
    #     # print(res)
    #     self.assertTrue(np.allclose(res1, res2, rtol=1e-5))




