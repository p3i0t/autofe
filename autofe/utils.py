import pandas as pd
from typing import List


# type aliases
FEATURE_TYPE  = 'feature'
INT_TYPE      = 'int'
FLOAT_TYPE    = 'float'


def compute_IC(
    df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    from_date: str = "20130101",
    to_date: str = "20211231",
):
    """Compute IC.
    Parameters
    ----------
    df : pd.DataFrame
        result pd.DataFrame contains columns including x_cols, y_col and ['symbol', 'date'].
    x_cols : List[str]
        list of names of left columns to compute IC.
    y_col : str
        name of right column to compute IC.
    from_date : str, optional
        start date, by default '20130101'
    to_date : str, optional
        end date, by default '20211231'
    Returns
    -------
    pd.DataFrame
        daily IC dataframe of given x_cols and y_col.
    """
    idx = (df["date"] >= from_date) & (df["date"] <= to_date)
    df_ = df.loc[idx, ["date", "symbol", y_col] + x_cols]
    IC = df_.groupby("date").apply(lambda x: x[x_cols].corrwith(x[y_col]))
    return IC