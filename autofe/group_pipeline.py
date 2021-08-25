import os
from collections import namedtuple
from typing import List
import warnings
import itertools
import logging
import pickle
from lightgbm.callback import early_stopping

import ray
import torch
import numpy as np
import pandas as pd
import lightgbm
from lightgbm import LGBMRegressor

from utils import compute_IC


# helper functions
def _reshape(xx: torch.Tensor, yy: torch.Tensor):
    # (date, time, symbols, n_factors) -> (date*symbols, time*n_factors)
    n_date, n_times, n_symbols, n_factors = xx.shape
    xx = xx.permute(0, 2, 1, 3).reshape(n_date*n_symbols, # pylint: disable=invalid-name
                                        n_times*n_factors).contiguous()
    yy = yy.reshape((-1,)) # pylint: disable=invalid-name
    return xx, yy # pylint: disable=invalid-name

def _filter(xx: torch.Tensor, yy: torch.Tensor, date_symbol_index: pd.MultiIndex): # pylint: disable=invalid-name
    "filter NaNs."
    valid_idx = ~yy.isnan() # pylint: disable=invalid-name

    return xx[valid_idx], yy[valid_idx], date_symbol_index[valid_idx.numpy()] # pylint: disable=invalid-name


DataSplit = namedtuple('DataSplit', field_names=['x', 'y', 'index', 'x_names'])


def _generate_dataset(x: torch.Tensor, y: torch.Tensor, dates: List[str],
                      symbols: List[str], x_names: List[str], train_valid12_test_ratios=(0.6, 0.1, 0.1, 0.2)):

    n_days = len(dates)
    a = int(sum(train_valid12_test_ratios[:1]) * n_days)
    b = int(sum(train_valid12_test_ratios[:2]) * n_days)
    c = int(sum(train_valid12_test_ratios[:3]) * n_days)

    train_x, train_y = _reshape(x[:a], y[:a])
    valid1_x, valid1_y = _reshape(x[a:b], y[a:b])
    valid2_x, valid2_y = _reshape(x[b:c], y[b:c])
    test_x, test_y = _reshape(x[c:], y[c:])

    train_index = pd.MultiIndex.from_product([dates[:a], symbols])
    valid1_index = pd.MultiIndex.from_product([dates[a:b], symbols])
    valid2_index = pd.MultiIndex.from_product([dates[b:c], symbols])
    test_index = pd.MultiIndex.from_product([dates[c:], symbols])

    train_x, train_y, train_index = _filter(train_x, train_y, train_index)
    train_set = DataSplit._make((train_x, train_y, train_index, x_names))

    valid1_x, valid1_y, valid1_index = _filter(valid1_x, valid1_y, valid1_index)
    valid1_set = DataSplit._make((valid1_x, valid1_y, valid1_index, x_names))

    valid2_x, valid2_y, valid2_index = _filter(valid2_x, valid2_y, valid2_index)
    valid2_set = DataSplit._make((valid2_x, valid2_y, valid2_index, x_names))

    test_x, test_y, test_index = _filter(test_x, test_y, test_index)
    test_set = DataSplit._make((test_x, test_y, test_index, x_names))
    return train_set, valid1_set, valid2_set, test_set


def _forward_inference(model, train_set, valid1_set, valid2_set, test_set):
    """Fit the model and inference on valid/test sets.
    """
    x, y = train_set.x.numpy(), train_set.y.numpy()
    valid1_x, valid1_y = valid1_set.x.numpy(), valid1_set.y.numpy()
    valid2_x, valid2_y = valid2_set.x.numpy(), valid2_set.y.numpy()
    test_x, test_y = test_set.x.numpy(), test_set.y.numpy()

    model.fit(x, y, feature_name=train_set.x_names,
              eval_set=[(valid1_x, valid1_y)],
              early_stopping_rounds=10, verbose=False)

    # valid & test
    pred = model.predict(valid1_x)
    df_valid1 = pd.DataFrame({'pred': pred, 'y': valid1_y}, index=valid1_set.index)
    df_valid1.index.names = ['date', 'symbol']
    df_valid1.reset_index(inplace=True)
    # valid_ic = compute_IC(df_valid, ['pred'], 'y').mean().iloc[0]

    pred = model.predict(valid2_x)
    df_valid2 = pd.DataFrame({'pred': pred, 'y': valid2_y}, index=valid2_set.index)
    df_valid2.index.names = ['date', 'symbol']
    df_valid2.reset_index(inplace=True)

    pred = model.predict(test_x)
    df_test = pd.DataFrame({'pred': pred, 'y': test_y}, index=test_set.index)
    df_test.index.names = ['date', 'symbol']
    df_test.reset_index(inplace=True)

    return model, df_valid1, df_valid2, df_test

ForwardArgs = namedtuple('ForwardArgs', field_names=['group_index', 'expressions', 'ic_threshold', 'topk', 'save_dir'])


@ray.remote
class GroupPipeline:
    def __init__(self, x: torch.Tensor, y: torch.Tensor, dates: List[str],
                 times: List[str], symbols: List[str], x_names: List[str], args) -> None:
        n_dates, n_times, n_symbols, n_factors = x.shape
        assert (len(dates) == n_dates and len(times) == n_times and
                len(symbols) == n_symbols and len(x_names) == n_factors), \
                'inconsistency of shape of x and given arguments.'
        assert y.shape == (n_dates, n_symbols), 'inconsistency between shapes of y and x.'
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)

        # initialize with raw input factors.
        self.x = x
        self.y = y
        self.dates = dates
        self.times = times
        self.symbols = symbols
        self.x_names = x_names
        self.args = args

    def forward(self, tuple_args):
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s] %(levelname)s (%(name)s) %(message)s')

        # Note that here we have to put `x` on GPU for very new group of expressions evaluation,
        # and then free the GPU memory due to the limited GPU memory.
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)
            x_cuda = torch.FloatTensor(self.x).cuda()

        # dict reference of the x_tensor.
        feature_tensor_dict = {self.x_names[i]: x_cuda[:, :, :, i]
                                    for i in range(len(self.x_names))}

        # Evaluation expressions.
        evaluated_expressions = [factor.eval(feature_tensor_dict).cpu()
                                 for factor in tuple_args.expressions]

        # clean GPU memory
        del feature_tensor_dict
        del x_cuda
        torch.cuda.empty_cache()
        # logging.info('expressions evaluation done.')

        factor_x = torch.cat([tensor.unsqueeze(dim=3) # pylint: disable=no-member
                              for tensor in evaluated_expressions], dim=3)
        renamed_factors_dict = {f'factor{idx:03d}': str(factor)
                                for idx, factor in enumerate(tuple_args.expressions)}
        renamed_full_names = self.x_names + list(renamed_factors_dict.keys())
        original_full_names = self.x_names + list(renamed_factors_dict.values())

        renamed_full_names_with_time = [s + '_' + t for t, s in
                                        itertools.product(self.times, renamed_full_names)]
        original_full_names_with_time = [s + '_' + t for t, s in
                                        itertools.product(self.times, original_full_names)]

        merge_x = torch.cat([self.x, factor_x], dim=3)
        train_set, valid1_set, valid2_set, test_set = _generate_dataset(
            merge_x, self.y, self.dates,
            self.symbols, renamed_full_names_with_time, self.args.train_valid12_test_ratios)

        model = LGBMRegressor(
            boosting_type='goss',
            learning_rate=0.01,
            importance_type='gain',
            n_jobs=self.args.pipeline_num_cpus,
            device='gpu',
            # gpu_device_id=gpu_id,
        )

        try:
            model, df_valid1, df_valid2, df_test = _forward_inference(model, train_set, valid1_set, valid2_set, test_set)
            valid1_ic = compute_IC(df_valid1, ['pred'], 'y').mean().iloc[0]
            valid2_ic = compute_IC(df_valid2, ['pred'], 'y').mean().iloc[0]
            test_ic = compute_IC(df_test, ['pred'], 'y').mean().iloc[0]
            logging.info(f"group {tuple_args.group_index}, valid1_IC: {valid1_ic:.4f}, valid2_ic: {valid2_ic:.4f}, test_IC: {test_ic:.4f}")
        except lightgbm.basic.LightGBMError as e:
            # Ignore a known exception, see https://github.com/microsoft/LightGBM/issues/3339
            if 'bin size 257 cannot run on GPU' in str(e):
                logging.info(f"A open and known LightGBMError on GPU catched. Give up this group and continue.")
                return
            else:
                # raise other exceptions.
                raise e

        if valid1_ic > tuple_args.ic_threshold:
            df_factor_importance = pd.DataFrame({
                'renamed_factor': renamed_full_names_with_time,
                'original_factor': original_full_names_with_time,
                'importance': list(model.feature_importances_)})

            # logging.info(f"start select top k factors.")
            df_topk = df_factor_importance.sort_values('importance', ascending=False)[:tuple_args.topk]
            topk_factor_time_names = [factor for factor in df_topk['renamed_factor'].to_list()
                                      if factor.startswith('factor')]

            topk_factors = []
            topk_expressions = []
            for factor in topk_factor_time_names:
                factor_str, time_str = factor.split('_')
                factor_idx = int(factor_str.lstrip('factor'))
                idx = self.times.index(time_str)

                factor_tensor = factor_x[:, :, :, factor_idx]
                topk_factors.append(factor_tensor[:, idx:idx+1, :])

                topk_expressions.append(tuple_args.expressions[factor_idx])

            # (n_date, topk, n_symbol)
            topk_array = torch.cat(topk_factors, dim=1).numpy()

            group_dir = f"{tuple_args.save_dir}/group{tuple_args.group_index}"
            if not os.path.exists(group_dir):
                os.makedirs(group_dir)

            df_factor_importance.to_csv(f'{group_dir}/df_importance.csv', index=False)
            # save results.
            np.save(f"{group_dir}/topk_array.npy", topk_array)
            with open(f'{group_dir}/topk_factor_names.txt', 'w') as outfile:
                for (name, expression) in zip(topk_factor_time_names, topk_expressions):
                    outfile.write(f"{name}, {expression}\n")

            expressions_set = set(topk_expressions)
            for idx, expression in enumerate(expressions_set):
                pickle.dump(factor, open(f"{group_dir}/expression{idx:04d}_{str(expression)[:200]}.pkl", 'wb'))


    def forward_baseline(self):
        """Fit and inference on raw input dataset.
        """
        assert torch.cuda.is_available()
        gpu_id = torch.cuda.current_device()
        model = LGBMRegressor(
            boosting_type='goss',
            learning_rate=0.01,
            importance_type='gain',
            n_jobs=self.args.pipeline_num_cpus,
            device='gpu',
            gpu_device_id=gpu_id,
        )
        full_names_with_time = [s + '_' + t for t, s in
                        itertools.product(self.times, self.x_names)]

        train_set, valid1_set, valid2_set, test_set = _generate_dataset(
            self.x, self.y, self.dates,
            self.symbols, full_names_with_time, self.args.train_valid12_test_ratios)
        return _forward_inference(model, train_set, valid1_set, valid2_set, test_set)


if __name__ == '__main__':
    pass



