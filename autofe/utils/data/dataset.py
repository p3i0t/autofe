from dataclasses import dataclass
from typing import List

import pandas as pd
import torch


# helper functions
def _reshape(xx: torch.Tensor, yy: torch.Tensor):
    # (date, time, symbols, n_factors) -> (date*symbols, time*n_factors)
    n_date, n_minutes, n_symbols, n_factors = xx.shape
    xx = xx.permute(0, 2, 1, 3).reshape(n_date*n_symbols, # pylint: disable=invalid-name
                                        n_minutes*n_factors).contiguous()
    # xx = np.nan_to_num(xx, nan=0, posinf=1e5, neginf=-1e5)
    # (n_date * n_symbols, )
    yy = yy.reshape((-1,)) # pylint: disable=invalid-name
    return xx, yy # pylint: disable=invalid-name

def _filter(xx: torch.Tensor, yy: torch.Tensor, date_symbol_index: pd.MultiIndex): # pylint: disable=invalid-name
    "filter NaNs."
    # assert isinstance(xx, torch.Tensor) and isinstance(yy, torch.Tensor)
    valid_idx = ~yy.isnan() # pylint: disable=invalid-name

    return xx[valid_idx], yy[valid_idx], date_symbol_index[valid_idx.numpy()] # pylint: disable=invalid-name



@dataclass
class TS4dDataset:
    """A dataset class representing the time-series ticker data in 4d (date, minute, symbol, factor) format.

    """
    x: torch.Tensor
    y: torch.Tensor
    dates: List[str]
    minites: List[str]
    symbols: List[str]
    factors: List[str]

    def _check_validation(self):
        n_dates, n_minutes, n_symbols, n_factors = self.x.shape
        assert self.y.shape == (n_dates * n_symbols,), 'y shape is not compatible to x'
        assert len(self.minutes) == n_minutes
        assert len(self.symbols) == n_symbols
        assert len(self.factors) == n_factors
        assert len(self.date) == n_dates

    def to_tabular(self):
        pass