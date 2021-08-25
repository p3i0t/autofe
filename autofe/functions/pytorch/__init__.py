import torch
from typing import Callable, List, Optional, Any
from ._nan_functions import (_nanargmax, _nanargmin, _nanmax, _nanmean,
                             _nanmin, _nanprod, _nanstd, _nanvar)

from .elementwise_functions import (_inverse, _negative, _protected_log,
                                    _protected_sqrt,_sigmoid, _tanh,
                                    _signedpower)

from .ts_functions import (_ts_day_corr, _ts_day_cov, _ts_day_prod,
                           _ts_day_rank, _ts_day_std, _ts_day_sum,
                           _ts_min_corr, _ts_min_cov, _ts_min_prod,
                           _ts_min_rank, _ts_min_std, _ts_min_sum,
                           _rank, _cross_rank, compute_cov,
                           compute_correlation)


FEATURE_TYPE  = 'feature'
DAY_INT_TYPE  = 'day_int'
MIN_INT_TYPE  = 'min_int'
FLOAT_TYPE    = 'float'

class _Function:
    def __init__(self,
                 func: Callable,
                 arity: int,
                 name: str,
                 argument_types: List[str],
                 is_ts: bool = None) -> None:
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

    def __str__(self) -> str:
        return self.name #+ f", arity={self.arity}"


# basic functions
add2 = _Function(func=torch.add, name='add', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE], is_ts=False)
sub2 = _Function(func=torch.sub, name='sub', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE], is_ts=False)
mul2 = _Function(func=torch.mul, name='mul', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE], is_ts=False)
div2 = _Function(func=torch.div, name='div', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE], is_ts=False)
sqrt1 = _Function(func=_protected_sqrt, name='sqrt', arity=1, argument_types=[FEATURE_TYPE], is_ts=False)
log1 = _Function(func=_protected_log, name='log', arity=1, argument_types=[FEATURE_TYPE], is_ts=False)
neg1 = _Function(func=_negative, name='neg', arity=1, argument_types=[FEATURE_TYPE], is_ts=False)
inv1 = _Function(func=_inverse, name='inv', arity=1, argument_types=[FEATURE_TYPE], is_ts=False)
abs1 = _Function(func=torch.abs, name='abs', arity=1, argument_types=[FEATURE_TYPE], is_ts=False)
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



_function_map = {
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
    # 'ts_day_prod': ts_day_prod2,
    # 'ts_min_prod': ts_min_prod2,
    'ts_day_std': ts_day_std2,
    'ts_min_std': ts_min_std2,
    # 'ts_day_rank': ts_day_rank2,  very memory-consuming
    # 'ts_min_rank': ts_min_rank2,  very memory-consuming
    # 'ts_min': ts_min2,
    # 'ts_max': ts_max2,
    # 'ts_argmin': ts_argmin2,
    # 'ts_argmax': ts_argmax2,
}



def compute_IC(x: torch.Tensor, y: torch.Tensor):
    nan_mask = x.isnan() | y.isnan()
    x[nan_mask] = float('nan')
    y[nan_mask] = float('nan')

    def dmean(x_in: torch.Tensor):
        return x_in.sub_(_nanmean(x_in, dim=2, keepdim=True))

    def normalize(x_in: torch.Tensor):
        norm = torch.nansum(x_in ** 2, dim=2, keepdim=True).sqrt()
        return x_in / norm
        # return (x_in - _nanmean(x_in, dim=2, keepdim=True)) / _nanstd(x_in, dim=2, keepdim=True)

    x_norm = normalize(dmean(x))
    y_norm = normalize(dmean(y))

    ICs = torch.nansum((x_norm * y_norm), dim=2)
    mean_IC = _nanmean(ICs)
    return mean_IC

def compute_rankIC(x: torch.Tensor, y: torch.Tensor):
    x_rank = _rank(x, dim=2)
    y_rank = _rank(y, dim=2)

    return compute_IC(x_rank, y_rank)

if __name__ == '__main__':
    pass
