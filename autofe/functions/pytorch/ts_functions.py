from typing import Tuple, Callable
import numpy as np
import torch
import torch.nn.functional as F
from ._nan_functions import (_nanargmax, _nanargmin, _nanmax, _nanmean,
                             _nanmin, _nanprod, _nanstd, _nanvar)

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
    assert len(x.shape) == 3
    return _rank(x, dim=2)


#########################
# rolling functions
#########################

def _rolling(x: torch.Tensor or Tuple[torch.Tensor, torch.Tensor], window:int, func: Callable) -> torch.Tensor:
    """Rolling always along the first axis and apply `func` on each window.
    The implementation is based on `unfolded` indices generated by `torch.Tensor.unfold`,
    instead of using naive for loop. The result is huge acceleration due to the
    parallelized computations in one go for all possible windows,
    while at the same time `window`-times of memory usuage. This function is
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

    def _pad(x_in: torch.Tensor, window: int, step:int =1) -> torch.Tensor:
        """Padding NaNs at the front.

        Parameters
        ----------
        x_in : torch.Tensor
            input to be rolled on.
        window : int
            sliding window size.
        step : int, optional
            step of sliding window, by default 1

        Returns
        -------
        torch.Tensor
            padded input.
        """
        padding = torch.zeros(window-step, *x_in.shape[1:]) + float('nan')
        return torch.cat([padding.to(x_in.device), x_in], dim=0)

    step = 1
    if isinstance(x, torch.Tensor):
        x = _pad(x, window=window, step=step)
        # print(f"x shape: {x.shape}")
        idx_unfold = torch.arange(x.shape[0]).unfold(dimension=0, size=window, step=step)
        # print(f"idx_unfold shape: {idx_unfold.shape}")
        res = func(x[idx_unfold])  # apply `func` on unfolded tenssor.
    elif isinstance(x, Tuple):
        # print('x shape', x[0].shape)
        x1, x2 = [_pad(ele, window=window, step=step) for ele in x]
        # print(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")
        idx_unfold = torch.arange(x1.shape[0]).unfold(dimension=0, size=window, step=step)
        # print(f"idx_unfold shape: {idx_unfold.shape}")
        # print(idx_unfold.shape)
        # print(idx_unfold)
        # print('x padding shape', x1.shape)
        # # print(x2.shape)

        # print('x unfold shape', x1[idx_unfold].shape)
        res = func(x1[idx_unfold], x2[idx_unfold])  # apply `func` on unfolded tenssor.
    else:
        raise Exception('Never should be here.')
    return res


def compute_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the correlation of the inputs x, y, who must be in the same shape.
    This function is not supposed to be called directly. All intermediate operations are applied on
    expanded dim=1.

    Parameters
    ----------
    x : torch.Tensor
        input x, 3D Tensor (date, d, minute, symbol) for date-rolling,
        or 2D Tensor (date*minite, d, symbol) for minute-rolling, where
        `d` is the expanded size of sliding window.
    y : torch.Tensor
        input y, 3D Tensor (date, d, minute, symbol) for date-rolling,
        or 2D Tensor (date*minite, d, symbol) for minute-rolling, where
        `d` is the expanded size of sliding window.
    """


    # assert x.shape == y.shape, 'x, y should be in the same shape.'
    # assert len(x.shape) in [2, 3], 'x should be 2D or 3D.'

    def _nan_dmean(x_in: torch.Tensor):
        return x_in - _nanmean(x_in, dim=1, keepdim=True)
        # return x_in - x_in.mean(dim=1, keepdim=True)

    def _nan_normalize(x_in: torch.Tensor):
        return x_in / (x_in ** 2).nansum(dim=1, keepdim=True).sqrt()
        # return x_in / x_in.std(dim=1, keepdim=True, unbiased=False)

    x_m, y_m = _nan_dmean(x), _nan_dmean(y)
    out_x = _nan_normalize(x_m)
    # print('x2 ', out_x)
    out_y = _nan_normalize(y_m)

    out_x.mul_(out_y)
    nonnan_idx = ~out_x.isnan()
    nonnan_cnt = nonnan_idx.sum(dim=1)
    r = out_x.nansum(dim=1)
    r[nonnan_cnt == 0] = float('nan')

    r.clamp_(min=-1.0, max=1.0)  # this is not correct, all nans will give 0 which should also be nan.
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
    inp = (x.view(-1, n_symbol), y.view(-1, n_symbol))
    res = _rolling(inp, window=d, func=compute_correlation)
    return res.view(n_date, n_min, n_symbol).contiguous()


def compute_cov(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the covariance of input tensors x, y, who are in the same shape.
    This function is not supposed to be called directly.
    All intermediate operations are applied on expanded dim=1.

    Parameters
    ----------
    x : torch.Tensor
        input x, 3D Tensor (date, d, minute, symbol) for date-rolling,
        or 2D Tensor (date*minite, d, symbol) for minute-rolling, where
        `d` is the expanded size of sliding window.
    y : torch.Tensor
        input y, 3D Tensor (date, d, minute, symbol) for date-rolling,
        or 2D Tensor (date*minite, d, symbol) for minute-rolling, where
        `d` is the expanded size of sliding window.
    Returns
    -------
    torch.Tensor
        output tensor
    """

    # assert x.shape == y.shape, 'x, y should be in the same shape.'
    # assert len(x.shape) in [2, 3], 'x should be 2D or 3D.'

    def dmean(x: torch.Tensor):
        return x - _nanmean(x, dim=1, keepdim=True)

    x_m, y_m = dmean(x), dmean(y)
    x_m.mul_(y_m)

    nonnan_idx = ~x_m.isnan()
    nonnan_cnt = nonnan_idx.sum(dim=1)
    res = x_m.nan_to_num_(nan=0.).sum(dim=1).div_(nonnan_cnt - 1)
    res[nonnan_cnt == 0] = float('nan') # all elements are nans, result should be nan.
    return res



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

    n_date, n_min, n_symbol = x.shape
    inp = (x.view(-1, n_symbol), y.view(-1, n_symbol))
    res = _rolling(inp, window=d, func=compute_cov)
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
        """Compute pct rank of x_i in the window.

        Parameters
        ----------
        x : torch.Tensor
            [description]

        Returns
        -------
        torch.Tensor
            [description]
        """
        return _rank(x, dim=1)[:, -1]

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
        return _rank(x, dim=1)[:, -1]

    n_date, n_min, n_symbol = x.shape
    inp = x.view(-1, n_symbol)
    res = _rolling(inp, window=d, func=adjusted_rank)
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
    return _rolling(x, window=d, func=lambda x_in: torch.nansum(x_in, dim=1, keepdim=False))


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
    n_date, n_min, n_symbol = x.shape
    inp = x.view(-1, n_symbol)
    res = _rolling(inp, window=d, func=lambda x_in: torch.nansum(x_in, dim=1, keepdim=False))
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
    return _rolling(x, window=d, func=lambda x_in: _nanprod(x_in, dim=1, keepdim=False))


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
    inp = x.view(-1, n_symbol)
    res = _rolling(inp, window=d, func=lambda x_in: _nanprod(x_in, dim=1, keepdim=False))
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
    return _rolling(x, window=d, func=lambda x_in: _nanstd(x_in, dim=1, keepdim=False))


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
    inp = x.view(-1, n_symbol)
    res = _rolling(inp, window=d, func=lambda x_in: _nanstd(x_in, dim=1, keepdim=False))
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
