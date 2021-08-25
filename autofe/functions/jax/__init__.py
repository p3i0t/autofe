from functools import partial
import jax
import jax.numpy as jnp
from typing import Callable, List, Optional, Any
from .elementwise_functions import (_inverse, _negative, _protected_log, 
                                    _protected_sqrt,_sigmoid, _tanh, 
                                    _signedpower, _replace_0_to_nan,
                                    _replace_inf_to_nan)

from .ts_functions import (_batched_corr, _batched_nancorr, _batched_nancov,
                           _ts_day_corr, _ts_day_cov, _ts_day_f, _ts_day_rank,
                           _ts_min_corr, _ts_min_cov, _ts_min_f, _ts_min_rank,
                           _rank, _cross_rank)



class _Function:
    def __init__(self, 
                 func: Callable, 
                 arity: int, 
                 name: str, 
                 argument_types: List[str], 
                 is_ts: bool) -> None:
        """Define a wrapper of function.

        Parameters
        ----------
        func : Callable
            function to be called.
        arity : int
            arity (number of arguments) of the function.
        name : str
            name of the function.
        argument_types : List[str]
            list of argument types, one of FEATURE_TYPE, INT_TYPE.
        is_ts : bool
            whether is time series (rolling) function.
        """        
        self.func = func
        self.arity = arity
        self.name = name
        self.argument_types = argument_types
        self.is_ts = is_ts
        
    def __call__(self, *args: Any) -> Any:
        return self.func(*args)

FEATURE_TYPE  = 'feature'
DAY_INT_TYPE  = 'day_int'
MIN_INT_TYPE  = 'min_int'
FLOAT_TYPE    = 'float'


# basic functions
add2 = _Function(func=jnp.add, name='add', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE], is_ts=False)
sub2 = _Function(func=jnp.subtract, name='sub', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE], is_ts=False)
mul2 = _Function(func=jnp.multiply, name='mul', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE], is_ts=False)
div2 = _Function(func=jnp.divide, name='div', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE], is_ts=False)
sqrt1 = _Function(func=_protected_sqrt, name='sqrt', arity=1, argument_types=[FEATURE_TYPE], is_ts=False)
log1 = _Function(func=_protected_log, name='log', arity=1, argument_types=[FEATURE_TYPE], is_ts=False)
neg1 = _Function(func=_negative, name='neg', arity=1, argument_types=[FEATURE_TYPE], is_ts=False)
inv1 = _Function(func=_inverse, name='inv', arity=1, argument_types=[FEATURE_TYPE], is_ts=False)
abs1 = _Function(func=jnp.abs, name='abs', arity=1, argument_types=[FEATURE_TYPE], is_ts=False)
# max2 = _Function(func=torch.Tensor.max, name='max', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE])
# min2 = _Function(func=torch.Tensor.min, name='min', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE])
# sin1 = _Function(func=np.sin, name='sin', arity=1, argument_types=[FEATURE_TYPE])
# cos1 = _Function(func=np.cos, name='cos', arity=1, argument_types=[FEATURE_TYPE])
# tan1 = _Function(func=np.tan, name='tan', arity=1, argument_types=[FEATURE_TYPE])
sigmoid1 = _Function(func=_sigmoid, name='sigmoid', arity=1, argument_types=[FEATURE_TYPE], is_ts=False)
tanh1 = _Function(func=_tanh, name='tanh', arity=1, argument_types=[FEATURE_TYPE], is_ts=False)
cross_rank1 = _Function(func=_cross_rank, name='cross_rank', arity=1, argument_types=[FEATURE_TYPE], is_ts=False)

ts_day_corr3 = _Function(func=_ts_day_corr, name='ts_day_corr', arity=3, argument_types=[FEATURE_TYPE, FEATURE_TYPE, DAY_INT_TYPE], is_ts=True)
ts_min_corr3 = _Function(func=_ts_min_corr, name='ts_min_corr', arity=3, argument_types=[FEATURE_TYPE, FEATURE_TYPE, MIN_INT_TYPE], is_ts=True)
ts_day_cov3 = _Function(func=_ts_day_cov, name='ts_day_cov', arity=3, argument_types=[FEATURE_TYPE, FEATURE_TYPE, DAY_INT_TYPE], is_ts=True)
ts_min_cov3 = _Function(func=_ts_min_cov, name='ts_min_cov', arity=3, argument_types=[FEATURE_TYPE, FEATURE_TYPE, MIN_INT_TYPE], is_ts=True)
ts_day_rank2 = _Function(func=_ts_day_rank, name='ts_day_rank', arity=2, argument_types=[FEATURE_TYPE, DAY_INT_TYPE], is_ts=True)
ts_min_rank2 = _Function(func=_ts_min_rank, name='ts_min_rank', arity=2, argument_types=[FEATURE_TYPE, MIN_INT_TYPE], is_ts=True)

_ts_day_sum = partial(_ts_day_f, jnp.nansum)
_ts_min_sum = partial(_ts_min_f, jnp.nansum)
_ts_day_prod = partial(_ts_day_f, jnp.nanprod)
_ts_min_prod = partial(_ts_min_f, jnp.nanprod)
_ts_day_std = partial(_ts_day_f, jnp.nanstd)
_ts_min_std = partial(_ts_min_f, jnp.nanstd)

ts_day_sum2 = _Function(func=_ts_day_sum, name='ts_day_sum', arity=2, argument_types=[FEATURE_TYPE, DAY_INT_TYPE], is_ts=True)
ts_min_sum2 = _Function(func=_ts_min_sum, name='ts_min_sum', arity=2, argument_types=[FEATURE_TYPE, MIN_INT_TYPE], is_ts=True)
ts_day_prod2 = _Function(func=_ts_day_prod, name='ts_day_prod', arity=2, argument_types=[FEATURE_TYPE, DAY_INT_TYPE], is_ts=True)
ts_min_prod2 = _Function(func=_ts_min_prod, name='ts_min_prod', arity=2, argument_types=[FEATURE_TYPE, MIN_INT_TYPE], is_ts=True)
ts_day_std2 = _Function(func=_ts_day_std, name='ts_day_std', arity=2, argument_types=[FEATURE_TYPE, DAY_INT_TYPE], is_ts=True)
ts_min_std2 = _Function(func=_ts_min_std, name='ts_min_std', arity=2, argument_types=[FEATURE_TYPE, MIN_INT_TYPE], is_ts=True)

# ts_min2 = _Function(func=_ts_min, name='ts_min', arity=2, argument_types=[FEATURE_TYPE, INT_TYPE])
# ts_max2 = _Function(func=_ts_max, name='ts_max', arity=2, argument_types=[FEATURE_TYPE, INT_TYPE])
# ts_argmin2 = _Function(func=_ts_argmin, name='ts_argmin', arity=2, argument_types=[FEATURE_TYPE, INT_TYPE])
# ts_argmax2 = _Function(func=_ts_argmax, name='ts_argmax', arity=2, argument_types=[FEATURE_TYPE, INT_TYPE])



function_map = {
    'add': add2,
    'sub': sub2,
    'mul': mul2,
    'div': div2,
    'sqrt': sqrt1,
    'log': log1,
    'abs': abs1,
    'inv': inv1,
    'neg': neg1,
    'sigmoid': sigmoid1,
    'tanh': tanh1,
    'cross_rank': cross_rank1,
    'ts_day_corr': ts_day_corr3,
    'ts_min_corr': ts_min_corr3,
    'ts_day_cov': ts_day_cov3,
    'ts_min_cov': ts_min_cov3,
    'ts_day_sum': ts_day_sum2,
    'ts_min_sum': ts_min_sum2,
    'ts_day_prod': ts_day_prod2,
    'ts_min_prod': ts_min_prod2,
    'ts_day_std': ts_day_std2,
    'ts_min_std': ts_min_std2,
    'ts_day_rank': ts_day_rank2,
    'ts_min_rank': ts_min_rank2,
    # 'ts_min': ts_min2,
    # 'ts_max': ts_max2,
    # 'ts_argmin': ts_argmin2,
    # 'ts_argmax': ts_argmax2,
}


jit_function_map = {
    'add': add2,
    'sub': sub2,
    'mul': mul2,
    'div': div2,
    'sqrt': jax.jit(sqrt1),
    'log': jax.jit(log1),
    'abs': abs1,
    'inv': jax.jit(inv1),
    'neg': jax.jit(neg1),
    'sigmoid': jax.jit(sigmoid1),
    'tanh': jax.jit(tanh1),
    'cross_rank': jax.jit(cross_rank1),
    'ts_day_corr': jax.jit(ts_day_corr3, static_argnums=2),
    'ts_min_corr': jax.jit(ts_min_corr3, static_argnums=2),
    'ts_day_cov': jax.jit(ts_day_cov3, static_argnums=2),
    'ts_min_cov': jax.jit(ts_min_cov3, static_argnums=2),
    'ts_day_sum': jax.jit(ts_day_sum2, static_argnums=1),
    'ts_min_sum': jax.jit(ts_min_sum2, static_argnums=1),
    'ts_day_prod': jax.jit(ts_day_prod2, static_argnums=1),
    'ts_min_prod': jax.jit(ts_min_prod2, static_argnums=1),
    'ts_day_std': jax.jit(ts_day_std2, static_argnums=1),
    'ts_min_std': jax.jit(ts_min_std2, static_argnums=1),
    'ts_day_rank': jax.jit(ts_day_rank2, static_argnums=1),
    'ts_min_rank': jax.jit(ts_min_rank2, static_argnums=1),
    # 'ts_min': ts_min2,
    # 'ts_max': ts_max2,
    # 'ts_argmin': ts_argmin2,
    # 'ts_argmax': ts_argmax2,
}



def compute_IC(x, y, mean=False):
    """Compute IC.

    Parameters
    ----------
    x : 2D Array.
        (date, symbol)
    y : 2D Array.
        (date, symbol)
    mean : bool, optional
        mean or not, by default False

    Returns
    -------
    1D Array or float
        Array of ICs or mean IC.
    """
    ICs = _batched_nancorr(x, y)
    if mean:
        return jnp.nanmean(ICs).item()
    else:
        return ICs


def compute_rankIC(x, y, mean=False):
    """Compute rank IC.

    Parameters
    ----------
    x : 2D Array.
        (date, symbol)
    y : 2D Array.
        (date, symbol)
    mean : bool, optional
        mean or not, by default False

    Returns
    -------
    1D Array or float
        Array of ICs or mean IC.
    """
    x_rank = _rank(x, axis=1)
    y_rank = _rank(y, axis=1)
    
    return compute_IC(x_rank, y_rank, mean=mean)    

if __name__ == '__main__':
    pass
    

