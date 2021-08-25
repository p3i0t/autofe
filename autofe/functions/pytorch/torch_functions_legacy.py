from typing import Callable, Any, List, Optional, Union, Tuple
import numpy as np
import torch
from torch.functional import norm
import torch.nn.functional as F
from utils import FEATURE_TYPE, INT_TYPE, FLOAT_TYPE
from ._nan_functions import (_nanmax, _nanmin, _nanargmax, _nanargmin, _nanmean, _nanprod,
                             _nanvar, _nanstd)

class _Function:
    def __init__(self, 
                 func: Callable, 
                 arity: int, 
                 name: str, 
                 argument_types: List[str], 
                 ts_level: Optional[str]=None) -> None:
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
        ts_level : Optional[str], optional
            Rolling granularity of the `func`, `day` or `min`. None if `func` is not a rolling function, by default None.
        """        
        self.func = func
        self.arity = arity
        self.name = name
        self.argument_types = argument_types
        self.ts_level = ts_level
        
    def __call__(self, *args: Any) -> Any:
        return self.func(*args)

# helper functions

def validate_input(x: torch.Tensor):
    assert len(x.shape) == 3, 'should be 3D torch.Tensor.'
    
def validate_inputs(x: torch.Tensor, y: torch.Tensor):
    assert len(x.shape) == 3, 'should be 3D torch.Tensor.'
    assert x.shape == y.shape, 'x, y should be in the same shape.'


#########################
# dimension-wise functions
#########################

def _rank(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute percent rank along a dimension. NaNs are allowed and ignored.

    Parameters
    ----------
    x : torch.Tensor
        

    Returns
    -------
    torch.Tensor
        return the percentage rank of dim=0, keep NaN values unchanged.
    """
    # validate_input(x)
    n = x.shape[dim]
    nan_idx = x.isnan()
    o = x.argsort(dim=dim).argsort(dim=dim).float() / n
    o[nan_idx] = np.NaN
    return o

def _cross_rank(x: torch.Tensor) -> torch.Tensor:
    """Cross-sectional percent rank.

    Parameters
    ----------
    x : torch.Tensor
        3D torch.Tensor

    Returns
    -------
    torch.Tensor
        output
    """    
    return _rank(x, dim=2)


#########################
# rolling functions
#########################

def _rolling_(x: torch.Tensor or Tuple[torch.Tensor, torch.Tensor], window:int, func: Callable) -> torch.Tensor:
    """Rolling along with the first axis. Assume the input x is stored in C-order (row-first), so that 
    we always roll along the first dimension, which is fastest due to the memory alignment. This function is 
    not supposed to be called directly.

    Parameters
    ----------
    x : torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        Each torch.Tensor be 3D Tensor (date, minute, symbol) for date-rolling 
        or 2D Tensor(date*minute, symbol) for minute-rolling.
    window : int
        size of the rolling window.
    func: Callable
        function to be called on the window.
    """
    
    cal_index = lambda ind: ind-window if ind - window >= 0 else 0  # compute left index
    if isinstance(x, torch.Tensor):
        # validate_input(x)
        n = x.shape[0]
        results = [func(x[cal_index(i):i]) for i in range(1, n+1)] 
    elif isinstance(x, Tuple):
        # validate_inputs(*x)
        n = x[0].shape[0]
        results = [func(x[0][cal_index(i):i], x[1][cal_index(i):i]) for i in range(1, n+1)] 
    else:
        raise Exception('never should be here.')
    return torch.stack(results)


def _rolling(x: torch.Tensor or Tuple[torch.Tensor, torch.Tensor], window:int, func: Callable) -> torch.Tensor:
    """Rolling along with the first axis. Assume the input x is stored in C-order (row-first), so that 
    we always roll along the first dimension, which is fastest due to the memory alignment. This function is 
    not supposed to be called directly.

    Parameters
    ----------
    x : torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        Each torch.Tensor be 3D Tensor (date, minute, symbol) for date-rolling 
        or 2D Tensor(date*minute, symbol) for minute-rolling.
    window : int
        size of the rolling window.
    func: Callable
        function to be called on the window.
    """
    
    cal_index = lambda ind: ind-window if ind - window >= 0 else 0  # compute left index
    if isinstance(x, torch.Tensor):
        # validate_input(x)
        n = x.shape[0]
        results = [func(x[cal_index(i):i]) for i in range(1, n+1)] 
    elif isinstance(x, Tuple):
        # validate_inputs(*x)
        n = x[0].shape[0]
        results = [func(x[0][cal_index(i):i], x[1][cal_index(i):i]) for i in range(1, n+1)] 
    else:
        raise Exception('never should be here.')
    return torch.stack(results)


def compute_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the correlation of the inputs x, y, who must be in the same shape.
    Both inputs are of window size `d` on the first dimensions. This function is not
    supposed to be called directly.

    Parameters
    ----------
    x : torch.Tensor
        input x, 3D Tensor (d, minute, symbol) for date-rolling,
        where `d` is the window size of `date` dimension; or 2D Tensor (d, symbol) for minute-rolling, where
        `d` is the collapsed dimension of `date` and `symbol`.
    y : torch.Tensor
        input x, 3D Tensor (d, minute, symbol) for date-rolling,
        where `d` is the window size of `date` dimension; or 2D Tensor (d, symbol) for minute-rolling, where
        `d` is the collapsed dimension of `date` and `symbol`.
    """
    
    assert x.shape == y.shape, 'x, y should be in the same shape.'
    assert len(x.shape) in [2, 3], 'x should be 2D or 3D.'
    
    def dmean(x: torch.Tensor):
        return x - x.mean(dim=0, keepdim=True)

    if x.shape[0] == 1:
        # special treatment for one element correlation, should be NaN if there is only one element.
        if len(x.shape) == 3:
            r = torch.zeros(x.shape[1], x.shape[2]).to(x.device) + np.NaN # set all NaN
        else:
            r = torch.zeros(x.shape[1]).to(x.device) + np.NaN # set all NaN
        return r
    
    x_m, y_m = dmean(x), dmean(y)
    out_x = F.normalize(x_m, p=2.0, dim=0)  # L2 normalization.
    out_y = F.normalize(y_m, p=2.0, dim=0)  # L2 normalization.
    # inner product
    r = out_x.mul_(out_y).sum(dim=0).clamp_(min=-1.0, max=1.0)
    return r


def _ts_day_corr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """Compute the correlation of inputs x, y over the past `d` days.

    Parameters
    ----------
    x : torch.Tensor
        input x, 3D Tensor (d, minute, symbol) for date-rolling,
        where `d` is the window size of `date` dimension;
    y : torch.Tensor
        input x, 3D Tensor (d, minute, symbol) for date-rolling,
        where `d` is the window size of `date` dimension;
    d : int
        sliding window size.

    Returns
    -------
    torch.Tensor
        output tensor
    """    
    validate_inputs(x, y)
    return _rolling((x, y), window=d, func=compute_correlation)


def _ts_min_corr(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """Compute the correlations of inputs x, y over the past `d` minutes.

    Parameters
    ----------
    x : torch.Tensor
        input x, 3D Tensor (date, minute, symbol), which would be collapsed 
        to (date*minute, symbol) for date-rolling.
    y : torch.Tensor
        input x, 3D Tensor (date, minute, symbol), which would be collapsed 
        to (date*minute, symbol) for date-rolling.
    d : int
        sliding window size.

    Returns
    -------
    torch.Tensor
        output tensor.
    """    
    # validate_inputs(x, y)
    
    n_date, n_min, n_symbol = x.shape
    res = _rolling((x.view(-1, n_symbol), y.view(-1, n_symbol)), window=d, func=compute_correlation)
    return res.view(n_date, n_min, n_symbol).contiguous()


def compute_cov(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the covariance of input tensors x, y, who are in the same shape.
    Both inputs are of window size `d` on the first dimensions. This function is not
    supposed to be called directly.

    Parameters
    ----------
    x : torch.Tensor
        input x, 3D Tensor (d, minute, symbol) for date-rolling,
        where `d` is the window size of `date` dimension; or 2D Tensor(d, symbol) for minute-rolling, where
        `d` is the collapsed dimension of `date` and `symbol`.
    y : torch.Tensor
        input x, 3D Tensor (d, minute, symbol) for date-rolling,
        where `d` is the window size of `date` dimension; or 2D Tensor(d, symbol) for minute-rolling, where
        `d` is the collapsed dimension of `date` and `symbol`.

    Returns
    -------
    torch.Tensor
        output tensor
    """
    
    assert x.shape == y.shape, 'x, y should be in the same shape.'
    assert len(x.shape) in [2, 3], 'x should be 2D or 3D.'
    
    def dmean(x: torch.Tensor):
        return x - x.mean(dim=0, keepdim=True)
    
    n = x.shape[0]
    
    x_m, y_m = dmean(x), dmean(y)
    return x_m.mul_(y_m).sum(dim=0) / (n-1)


def _ts_day_cov(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """Compute the covariance of the inputs x, y over the past `d` days. 

    Parameters
    ----------
    x : torch.Tensor
        3D Tensor (date, minute, symbol) for date-rolling 
        or 2D Tensor(date*minute, symbol) for minute-rolling.
    y : torch.Tensor
        3D Tensor (date, minute, symbol) for date-rolling 
        or 2D Tensor(date*minute, symbol) for minute-rolling.
    d : int
        sliding window size.

    Returns
    -------
    torch.Tensor
        resulting Tensor of rolling covariance.
    """
    validate_inputs(x, y)
    return _rolling((x, y), window=d, func=compute_cov)


def _ts_min_cov(x: torch.Tensor, y: torch.Tensor, d: int) -> torch.Tensor:
    """Compute the covariance of inputs x, y, over the past `d` minutes.

    Parameters
    ----------
    x : torch.Tensor
        3D Tensor (date, minute, symbol), which would be clapsed to 
        2D Tensor (date*minute, symbol) for minute-rolling.
    y : torch.Tensor
        3D Tensor (date, minute, symbol), which would be clapsed to 
        2D Tensor (date*minute, symbol) for minute-rolling.
    d : int
        sliding window size.

    Returns
    -------
    torch.Tensor
        resulting Tensor of rolling covariance.
    """   
    validate_inputs(x, y)
    
    n_date, n_min, n_symbol = x.shape
    res = _rolling((x.view(-1, n_symbol), y.view(-1, n_symbol)), window=d, func=compute_cov)
    return res.view(n_date, n_min, n_symbol).contiguous()


def _ts_day_rank(x: torch.Tensor, d: int) -> torch.Tensor:  # slow
    """Compute the percent rank of the input x over the past `d` days.

    Parameters
    ----------
    x : torch.Tensor
        3D Tensor (date, minute, symbol) for date-rolling.
    d : int
        sliding window size.

    Returns
    -------
    torch.Tensor
        output tensor
    """    
    # helper function
    def adjusted_rank(x: torch.Tensor) -> torch.Tensor:
        return _rank(x, dim=0)[-1]

    return _rolling(x, window=d, func=adjusted_rank)
    

def _ts_min_rank(x: torch.Tensor, d: int) -> torch.Tensor:  # slow
    """Compute the percent rank of the input x over the past `d` minutes.

    Parameters
    ----------
    x : torch.Tensor
        3D Tensor (date, minute, symbol), which would be collapsed to 
        2D Tensor (date*minute, symbol) for minute-rolling.
    d : int
        sliding window size.

    Returns
    -------
    torch.Tensor
        output tensor
    """
  
    # helper function
    def adjusted_rank(x: torch.Tensor) -> torch.Tensor:
        return _rank(x, dim=0)[-1]

    n_date, n_min, n_symbol = x.shape
    res = _rolling(x.view(-1, n_symbol), window=d, func=adjusted_rank)
    return res.view(n_date, n_min, n_symbol).contiguous()


def _ts_day_sum(x: torch.Tensor, d: int) -> torch.Tensor:
    """compute the sumation of the input x over the `d` days.

    Parameters
    ----------
    x : torch.Tensor
        3D Tensor (date, minute, symbol) for date-rolling.
    d : int
        sliding window size.

    Returns
    -------
    torch.Tensor
        output tensor.
    """
    return _rolling(x, window=d, func=lambda x_in: torch.nansum(x_in, dim=0, keepdim=False)) 


def _ts_min_sum(x: torch.Tensor, d: int) -> torch.Tensor:
    """compute the sumation of the input x over the `d` minutes.

    Parameters
    ----------
    x : torch.Tensor
        3D Tensor (date, minute, symbol), which would be collapsed to 
        2D Tensor (date*minute, symbol) for minute-rolling.
    d : int
        sliding window size.

    Returns
    -------
    torch.Tensor
        output tensor.
    """
    # validate_input(x)
    n_date, n_min, n_symbol = x.shape
    res = _rolling(x.view(-1, n_symbol), window=d, func=lambda x_in: torch.nansum(x_in, dim=0, keepdim=False))
    return res.view(n_date, n_min, n_symbol).contiguous()


def _ts_day_prod(x: torch.Tensor, d: int) -> torch.Tensor:
    """compute the product of the input x over the past `d` days.

    Parameters
    ----------
    x : torch.Tensor
        3D Tensor (date, minute, symbol) for date-rolling.
    d : int
        sliding window size.

    Returns
    -------
    torch.Tensor
        output tensor
    """
    # validate_input(x)
    return _rolling(x, window=d, func=lambda x_in: _nanprod(x_in, dim=0, keepdim=False))


def _ts_min_prod(x: torch.Tensor, d: int) -> torch.Tensor:
    """compute the product of the input x over the past `d` minutes.

    Parameters
    ----------
    x : torch.Tensor
        3D Tensor (date, minute, symbol), which would be collapsed to 
        2D Tensor (date*minute, symbol) for minute-rolling.
    d : int
        sliding window size.

    Returns
    -------
    torch.Tensor
        output tensor
    """
    # validate_input(x)
    n_date, n_min, n_symbol = x.shape
    res = _rolling(x.view(-1, n_symbol), window=d, func=lambda x_in: _nanprod(x_in, dim=0, keepdim=False))
    return res.view(n_date, n_min, n_symbol).contiguous()

    
def _ts_day_std(x: torch.Tensor, d: int) -> torch.Tensor:
    """Compute the std of the input x over the past `d` days.

    Parameters
    ----------
    x : torch.Tensor
        3D Tensor (date, minute, symbol) for date-rolling.
    d : int
        sliding window size.

    Returns
    -------
    torch.Tensor
        output tensor
    """
    return _rolling(x, window=d, func=lambda x_in: _nanstd(x_in, dim=0, keepdim=False))


def _ts_min_std(x: torch.Tensor, d: int) -> torch.Tensor:
    """Compute the std of the input x over the past `d` minutes.

    Parameters
    ----------
    x : torch.Tensor
        3D Tensor (date, minute, symbol), which would be collapsed to 
        2D Tensor (date*minute, symbol) for minute-rolling.
    d : int
        sliding window size.

    Returns
    -------
    torch.Tensor
        output tensor
    """
    # validate_input(x)
    
    n_date, n_min, n_symbol = x.shape
    res = _rolling(x.view(-1, n_symbol).contiguous(), window=d, func=lambda x_in: _nanstd(x_in, dim=0, keepdim=False))
    return res.view(n_date, n_min, n_symbol).contiguous()


# TODO: to be finished.
def _ts_min(x: torch.Tensor, d: int) -> torch.Tensor:
    """compute the minimum of the factor of particular `symbol` over the past `d` time units.

    Parameters
    ----------
    x : torch.Tensor
        3D Tensor (datetime-like rolling dimension, factor, symbol), factor dim is 1 by default.
    d : int
        size of the sliding window.

    Returns
    -------
    torch.Tensor
        output tensor
    """ 
    return _rolling(x, window=d, func=lambda x_in: _nanmin(x_in, dim=0, keepdim=True)[0])


def _ts_argmin(x: torch.Tensor, d: int) -> torch.Tensor:
    return _rolling(x, window=d, func=lambda x_in: _nanargmin(x_in, dim=0, keepdim=True))
    # return _rolling(x, window=d, func=lambda x_in: x_in.min(dim=0, keepdim=True)[1])

def _ts_max(x: torch.Tensor, d: int) -> torch.Tensor:
    """compute the maximum of the factor of particular `symbol` over the past `d` time units.

    Parameters
    ----------
    x : torch.Tensor
        3D Tensor (datetime-like rolling dimension, factor, symbol), factor dim is 1 by default.
    d : int
        size of the sliding window.

    Returns
    -------
    torch.Tensor
        output tensor
    """ 
    return _rolling(x, window=d, func=lambda x_in: _nanmax(x_in, dim=0, keepdim=True)[0])

    
def _ts_argmax(x: torch.Tensor, d: int) -> torch.Tensor:
    return _rolling(x, window=d, func=lambda x_in: _nanargmax(x_in, dim=0, keepdim=True))   



# basic functions
add2 = _Function(func=torch.add, name='add', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE])
sub2 = _Function(func=torch.sub, name='sub', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE])
mul2 = _Function(func=torch.mul, name='mul', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE])
div2 = _Function(func=torch.div, name='div', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE])
sqrt1 = _Function(func=_protected_sqrt, name='sqrt', arity=1, argument_types=[FEATURE_TYPE])
log1 = _Function(func=_protected_log, name='log', arity=1, argument_types=[FEATURE_TYPE])
neg1 = _Function(func=_negative, name='neg', arity=1, argument_types=[FEATURE_TYPE])
inv1 = _Function(func=_inverse, name='inv', arity=1, argument_types=[FEATURE_TYPE])
abs1 = _Function(func=torch.abs, name='abs', arity=1, argument_types=[FEATURE_TYPE])

# max2 = _Function(func=torch.Tensor.max, name='max', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE])
# min2 = _Function(func=torch.Tensor.min, name='min', arity=2, argument_types=[FEATURE_TYPE, FEATURE_TYPE])
# sin1 = _Function(func=np.sin, name='sin', arity=1, argument_types=[FEATURE_TYPE])
# cos1 = _Function(func=np.cos, name='cos', arity=1, argument_types=[FEATURE_TYPE])
# tan1 = _Function(func=np.tan, name='tan', arity=1, argument_types=[FEATURE_TYPE])
sigmoid1 = _Function(func=_sigmoid, name='sigmoid', arity=1, argument_types=[FEATURE_TYPE])
tanh1 = _Function(func=_tanh, name='tanh', arity=1, argument_types=[FEATURE_TYPE])
cross_rank1 = _Function(func=_cross_rank, name='cross_rank', arity=1, argument_types=[FEATURE_TYPE])
ts_day_corr3 = _Function(func=_ts_day_corr, name='ts_day_corr', arity=3, argument_types=[FEATURE_TYPE, FEATURE_TYPE, INT_TYPE])
ts_min_corr3 = _Function(func=_ts_min_corr, name='ts_min_corr', arity=3, argument_types=[FEATURE_TYPE, FEATURE_TYPE, INT_TYPE])
ts_day_cov3 = _Function(func=_ts_day_cov, name='ts_day_cov', arity=3, argument_types=[FEATURE_TYPE, FEATURE_TYPE, INT_TYPE])
ts_min_cov3 = _Function(func=_ts_min_cov, name='ts_min_cov', arity=3, argument_types=[FEATURE_TYPE, FEATURE_TYPE, INT_TYPE])

ts_day_rank2 = _Function(func=_ts_day_rank, name='ts_day_rank', arity=2, argument_types=[FEATURE_TYPE, INT_TYPE])
ts_min_rank2 = _Function(func=_ts_min_rank, name='ts_min_rank', arity=2, argument_types=[FEATURE_TYPE, INT_TYPE])
ts_day_sum2 = _Function(func=_ts_day_sum, name='ts_day_sum', arity=2, argument_types=[FEATURE_TYPE, INT_TYPE])
ts_min_sum2 = _Function(func=_ts_min_sum, name='ts_min_sum', arity=2, argument_types=[FEATURE_TYPE, INT_TYPE])
ts_day_prod2 = _Function(func=_ts_day_prod, name='ts_day_prod', arity=2, argument_types=[FEATURE_TYPE, INT_TYPE])
ts_min_prod2 = _Function(func=_ts_min_prod, name='ts_min_prod', arity=2, argument_types=[FEATURE_TYPE, INT_TYPE])
ts_day_std2 = _Function(func=_ts_day_std, name='ts_day_std', arity=2, argument_types=[FEATURE_TYPE, INT_TYPE])
ts_min_std2 = _Function(func=_ts_min_std, name='ts_min_std', arity=2, argument_types=[FEATURE_TYPE, INT_TYPE])

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
    'ts_day_rank': ts_day_rank2,
    'ts_min_rank': ts_min_rank2,
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
    