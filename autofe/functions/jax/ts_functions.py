from typing import Tuple, Callable
import jax
import jax.numpy as jnp
from .elementwise_functions import _replace_0_to_nan, _replace_inf_to_nan


#########################
# dimension-wise functions
#########################

def _rank(x, axis: int):
    """Compute percent rank along a dimension. NaNs are allowed and ignored.
    """
    n = x.shape[axis]
    o = jnp.argsort(jnp.argsort(x, axis=axis), axis=axis) / n
    return jnp.where(jnp.isnan(x), float('nan'), o)


def _cross_rank(x):
    """Cross-sectional percent rank.
    """
    assert len(x.shape) == 3
    return _rank(x, dim=2)


#########################
# rolling functions
#########################

def _pad(x, d: int):
    padding = jnp.zeros((d, *x.shape[1:])) + float('nan')
    return jnp.concatenate([padding, x], axis=0)

def _ts_day_f(f: Callable, x, d: int):
    """Compute the sumation of the input x over the `d` days. `f` is a (nan-aware) function
    in jax.numpy, e.g. jnp.nansum, jnp.nanprod or jnp.nanmean.
    """
    n = x.shape[0]
    idx_unfold = jnp.arange(n)[:, None] + jnp.arange(d)
    
    x = _pad(x, d-1)
    x = f(x[idx_unfold, ...], axis=1, keepdims=False)
    return _replace_0_to_nan(x)

def _ts_min_f(f: Callable, x, d: int):
    """Compute the `f` of the input x over the `d` minutes. `f` is a (nan-aware) function
    in jax.numpy, e.g. jnp.nansum, jnp.nanprod. 
    """
    n_date, n_min, n_symbol = x.shape
    x = x.reshape(-1, n_symbol)
    idx_unfold = jnp.arange(x.shape[0])[:, None] + jnp.arange(d)
    x = _pad(x, d-1)
    
    x = f(x[idx_unfold, ...], axis=1, keepdims=False).reshape(n_date, n_min, n_symbol)
    return _replace_0_to_nan(x)


def _z_score(x):
    """Compute (nan-aware) `z`-score. This function is not supposed 
    to be called directly.
    """
    x_mean = jnp.nanmean(x, axis=1)
    x_std = jnp.nanstd(x, axis=1)
    return (x - x_mean) / (x_std + 1e-6)


_corr = lambda x, y: jnp.correlate(x, y)


def _nancorr(x, y):
    """Compute the (nan-aware) correlation of two 1D arrays. 
    This function is not supposed to be called directly.
    """
    def _dmean(x_in):
        return x_in - jnp.nanmean(x_in)
    
    def _normalize(x_in):
        return x_in / jnp.sqrt(jnp.nansum(x_in ** 2))
    
    element_prod = _normalize(_dmean(x)) * _normalize(_dmean(y))
    r = jnp.nansum(element_prod)
    # all nans will generate 0, so replace back to nan.
    return _replace_0_to_nan(jnp.clip(r, -1.0, 1.0))


def _nancov(x, y):
    """Compute nan-aware covariance of two 1-D arrays.
    This function is not supposed to be called directly.
    """
    def _dmean(x_in):
        return x_in - jnp.nanmean(x_in)
    
    element_prod = _dmean(x) * _dmean(y)
    nonnan_cnt = jnp.sum(~jnp.isnan(element_prod))
    out = jnp.nansum(element_prod) / (nonnan_cnt - 1)
    # all nans will generate 0, so replace back to nan.
    return _replace_0_to_nan(out)

_batched_corr = jax.vmap(_corr, in_axes=0, out_axes=0) # axis=0 is the batch dim, and keep it in the output.
_batched_nancorr = jax.vmap(_nancorr, in_axes=0, out_axes=0) # axis=0 is the batch dim, and keep it in the output.
_batched_nancov = jax.vmap(_nancov, in_axes=0, out_axes=0) # axis=0 is the batch dim, and keep it in the output.


def _ts_day_corr(x, y, d: int):
    """compute the correlations of x, y over the `d` days.
    """
    n_date, n_min, n_symbol = x.shape
    idx_unfold = jnp.arange(x.shape[0])[:, None] + jnp.arange(d)
    
    x = _pad(x, d-1)
    y = _pad(y, d-1)
    x_in = x[idx_unfold, ...] # (n_date, d, n_min, n_symbol) 
    y_in = y[idx_unfold, ...] # (n_date, d, n_min, n_symbol) 
    x_in = jnp.transpose(x_in, (0, 2, 3, 1)).reshape(-1, d)
    y_in = jnp.transpose(y_in, (0, 2, 3, 1)).reshape(-1, d)
    return _batched_nancorr(x_in, y_in).reshape(n_date, n_min, n_symbol)


def _ts_min_corr(x, y, d: int):
    """compute the correlations of x, y over the `d` minutes.
    """
    n_date, n_min, n_symbol = x.shape
    x = x.reshape(-1, n_symbol)
    y = y.reshape(-1, n_symbol)
    idx_unfold = jnp.arange(x.shape[0])[:, None] + jnp.arange(d)
    
    x = _pad(x, d-1)
    y = _pad(y, d-1)
    x_in = x[idx_unfold, ...] # (n_date*n_min, d, n_symbol) 
    y_in = y[idx_unfold, ...] # (n_date*n_min, d, n_symbol) 
    x_in = jnp.transpose(x_in, (0, 2, 1)).reshape(-1, d)
    y_in = jnp.transpose(y_in, (0, 2, 1)).reshape(-1, d)
    
    return _batched_nancorr(x_in, y_in).reshape(n_date, n_min, n_symbol)


def _ts_day_cov(x, y, d: int):
    """compute the covariances of x, y over the `d` days.
    """
    n_date, n_min, n_symbol = x.shape
    idx_unfold = jnp.arange(x.shape[0])[:, None] + jnp.arange(d)
    
    x = _pad(x, d-1)
    y = _pad(y, d-1)
    x_in = x[idx_unfold, ...] # (n_date, d, n_min, n_symbol) 
    y_in = y[idx_unfold, ...] # (n_date, d, n_min, n_symbol) 
    x_in = jnp.transpose(x_in, (0, 2, 3, 1)).reshape(-1, d)
    y_in = jnp.transpose(y_in, (0, 2, 3, 1)).reshape(-1, d)
    return _batched_nancov(x_in, y_in).reshape(n_date, n_min, n_symbol)


def _ts_min_cov(x, y, d: int):
    """compute the covariances of x, y over the `d` minutes.
    """
    n_date, n_min, n_symbol = x.shape
    x = x.reshape(-1, n_symbol)
    y = y.reshape(-1, n_symbol)
    idx_unfold = jnp.arange(x.shape[0])[:, None] + jnp.arange(d)
    
    x = _pad(x, d-1)
    y = _pad(y, d-1)
    x_in = x[idx_unfold, ...] # (n_date*n_min, d, n_symbol) 
    y_in = y[idx_unfold, ...] # (n_date*n_min, d, n_symbol) 
    x_in = jnp.transpose(x_in, (0, 2, 1)).reshape(-1, d)
    y_in = jnp.transpose(y_in, (0, 2, 1)).reshape(-1, d)
    
    return _batched_nancov(x_in, y_in).reshape(n_date, n_min, n_symbol)


def _ts_day_rank(x, d: int):  # slow
    """Compute the percent rank of the input x over the past `d` days.
    """    
    idx_unfold = jnp.arange(x.shape[0])[:, None] + jnp.arange(d)
    x = _pad(x, d-1)
    x_in = x[idx_unfold, ...] # (n_date, d, n_min, n_symbol) 
    return _rank(x_in, axis=1)[:, -1]


def _ts_min_rank(x, d: int):  # slow
    """Compute the percent rank of the input x over the past `d` minites.
    """
    n_date, n_min, n_symbol = x.shape
    x = x.reshape(-1, n_symbol) # (n_date*n_min, n_symbol)
    
    idx_unfold = jnp.arange(x.shape[0])[:, None] + jnp.arange(d)
    x = _pad(x, d-1)
    x_in = x[idx_unfold, ...] # (n_date*n_min, d, n_symbol) 
    return _rank(x_in, axis=1)[:, -1].reshape(n_date, n_min, n_symbol)
    

if __name__ == '__main__':
    n = 10
    p = 3
    q = 5
    d = 2
    x = jnp.arange(n*p, dtype='float32').reshape(n, p)
    x = x.at[0].add(float('nan'))
    x = x.at[1].add(float('nan'))
    x = x.at[2].add(float('nan'))
    
    print(x, x.shape)
    res = _ts_day_f(jnp.nansum, x, 3)
    print('xx', res, res.shape)

