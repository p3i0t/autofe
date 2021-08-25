from typing import Optional
import torch

def _nanmax(x: torch.Tensor, dim: Optional[int]=None, keepdim: Optional[bool]=False):
    """Same as torch.max(), except for ignoring the possible NaNs.
    """
    return torch.nan_to_num_(x, nan=-float('inf')).max(dim=dim, keepdim=keepdim)[0]


def _nanargmax(x: torch.Tensor, dim: Optional[int]=None, keepdim: Optional[bool]=False):
    """Same as torch.argmax(), except for ignoring the possible NaNs.
    """
    return torch.nan_to_num_(x, nan=-float('inf')).argmax(dim=dim, keepdim=keepdim)


def _nanmin(x: torch.Tensor, dim: Optional[int]=None, keepdim: Optional[bool]=False):
    """Same as torch.min(), except for ignoring the possible NaNs.
    """
    return torch.nan_to_num_(x, nan=float('inf')).min(dim=dim, keepdim=keepdim)[0]


def _nanargmin(x: torch.Tensor, dim: Optional[int]=None, keepdim: Optional[bool]=False):
    """Same as torch.argmin(), except for ignoring the possible NaNs.
    """
    return torch.nan_to_num_(x, nan=float('inf')).argmin(dim=dim, keepdim=keepdim)


def _nanmean(x: torch.Tensor, dim: Optional[int]=None, keepdim: Optional[bool]=False):
    """Same as torch.mean(), except for ignoring the possible NaNs.
    """
    nonnan_idx = ~x.isnan()
    if dim is None:
        dim = list(range(x.ndim))
    return torch.nansum(x, dim=dim, keepdim=keepdim) / torch.nansum(nonnan_idx, dim=dim, keepdim=keepdim).float()


def _nanprod(x: torch.Tensor, dim: Optional[int]=None, keepdim: Optional[bool]=False):
    """Same as torch.prod(), except for replacing the possible NaNs with 1.0.
    """
    return torch.nan_to_num_(x, nan=1.0).prod(dim=dim, keepdim=keepdim)


def _nanvar(x: torch.Tensor, dim: Optional[int]=None, keepdim: Optional[bool]=False, ddof:int=0):
    """Same as torch.var(), except for ignoring the possible NaNs.
    """
    nonnan_idx = ~x.isnan()
    nonnan_count = torch.nansum(nonnan_idx, dim=dim, keepdim=True).float()
    nan_mean = torch.nansum(x, dim=dim, keepdim=True) / nonnan_count
    
    var = ((x - nan_mean)**2).nansum(dim=dim, keepdim=True) / (nonnan_count - ddof)
    if keepdim is False:
        var = torch.squeeze(var, dim=dim)
    return var


def _nanstd(x: torch.Tensor, dim: Optional[int]=None, keepdim: Optional[bool]=False, ddof:int=0):
    """Same as torch.std(), except for ignoring the possible NaNs.
    """
    var = _nanvar(x, dim=dim, keepdim=keepdim, ddof=ddof)
    return var.sqrt_()
