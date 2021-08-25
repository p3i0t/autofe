import logging
import os
import random
import time
import pickle
import warnings
from typing import List, Tuple

import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import ray
from ray.util import ActorPool
import torch
from lightgbm import LGBMRegressor

from pytorch_functions import FEATURE_TYPE, DAY_INT_TYPE, MIN_INT_TYPE, _function_map
from group_pipeline import GroupPipeline, ForwardArgs, compute_IC
from expression_tree import ExpressionTree, ToolBox


logger = logging.getLogger(__name__)

def build_toolbox(args, input_factors):
    function_set = list(_function_map.values())
    tb = ToolBox(function_set)
    # maintain a feature map & keep the len of factor as small as possible.
    # feature_map = {f"f{idx}": factor for idx, factor in enumerate(input_factors)}
    tb.register_type('feature', input_factors)
    tb.register_type(DAY_INT_TYPE, args.day_interval_list)
    tb.register_type(MIN_INT_TYPE, args.min_interval_list)


def init_population(toolbox: ToolBox, expression_depth: Tuple[int, int],
                    n_expressions: int, save_dir: str) -> None:
    """Init the population of generate random expressions. Save each of them in pickle format.
    The format of each ExpressionTree's file name is `factor000xx_{expression_name}`. We use 5 the
    width of the int index padded with prefix 0s. The `expression_name` is the recursive
    string format of the ExpressionTree.

    Parameters
    ----------
    toolbox : ToolBox
        object with input names, day-level and minute-level candidate lists.
    n_expressions : int, optional
        number of factors to generate, by default 2000
    expression_depth : tuple, optional
        depth tuple of (low, high)
    save_dir: str
        save directory of the randomly generated ExpressionTree objects.
    """

    if os.path.exists(save_dir):
        factor_files = [file for file in os.listdir(save_dir)
                        if file.endswith('.pkl') and file.startswith('factor')]
        if len(factor_files) == n_expressions:
            logger.info(f'{n_expressions} expressions already generated. Launching evaluation.')
            return

    s = time.time()
    factors_list = []
    for _ in range(n_expressions):
        t = ExpressionTree(
            node_type=FEATURE_TYPE,
            toolbox=toolbox,
            expression_depth=expression_depth
            )
        t.generate_tree(grow=True, max_depth=expression_depth[1], cur_depth=0)

        factors_list.append(t)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    t = time.time()
    logger.info(f'generated {n_expressions} random expressions, time elapsed: {t-s:.4f}')

    logger.info(f'save to directory: {save_dir}')
    with open(f'{save_dir}/factor_names.txt', 'w') as outfile:
        for idx, factor in enumerate(factors_list):
            outfile.write(f"factor{idx:05d}_{str(factor)[:200]}\n")
            # there is a length limit for filename in linux system
            pickle.dump(factor, open(f"{save_dir}/factor{idx:04d}_{str(factor)[:200]}.pkl", 'wb'))
    outfile.close()



@ray.remote
def eval_group_topk(group_index, topk_x, y, dates, symbols,
                    x_names: List[str],
                    train_valid12_test_ratios=(0.6, 0.1, 0.1, 0.2),
                    importance_percentile=0.8):

    # print(f"hello group: {group_index}")
    n_days = len(dates)
    a = int(sum(train_valid12_test_ratios[:1]) * n_days)
    b = int(sum(train_valid12_test_ratios[:2]) * n_days)
    c = int(sum(train_valid12_test_ratios[:3]) * n_days)

    train_topk_x, train_y = topk_x[:a], y[:a]
    valid1_topk_x, valid1_y = topk_x[a:b], y[a:b]
    valid2_topk_x, valid2_y = topk_x[b:c], y[b:c]
    test_topk_x, test_y = topk_x[c:], y[c:]

    train_y = np.reshape(train_y, (-1,))
    valid1_y = np.reshape(valid1_y, (-1,))
    valid2_y = np.reshape(valid2_y, (-1,))
    test_y = np.reshape(test_y, (-1,))

    def _filter(xx, yy):
        valid_idx = ~np.isnan(yy)

        xx = xx[valid_idx]
        yy = yy[valid_idx]
        return xx, yy

    _, n_factor, _ = topk_x.shape
    train_topk_x = np.reshape(np.transpose(train_topk_x, (0, 2, 1)), (-1, n_factor))
    valid1_topk_x = np.reshape(np.transpose(valid1_topk_x, (0, 2, 1)), (-1, n_factor))
    valid2_topk_x = np.reshape(np.transpose(valid2_topk_x, (0, 2, 1)), (-1, n_factor))
    test_topk_x = np.reshape(np.transpose(test_topk_x, (0, 2, 1)), (-1, n_factor))

    model = LGBMRegressor(
        boosting_type='goss',
        learning_rate=0.01,
        importance_type='gain',
        device='gpu',
        n_jobs=8,
    )

    renames = [f"f{idx}" for idx, _ in enumerate(x_names)]

    train_topk_x_, train_y_ = _filter(train_topk_x, train_y)
    model.fit(train_topk_x_, train_y_,
              feature_name=renames, verbose=False, eval_set=[(valid2_topk_x, valid2_y)])

    # df_train = pd.DataFrame(data=train_topk_x, columns=x_names,
    #                         index=pd.MultiIndex.from_product([dates[:a], symbols]))
    # df_train.index.names = ['date', 'symbol']
    # df_train['y'] = train_y
    # df_train.reset_index(inplace=True)


    df = pd.DataFrame({
        'renamed_factor': renames,
        'original_factor': x_names,
        'importance': list(model.feature_importances_)})

    df = df.sort_values('importance', ascending=False)
    df['importance'] = df['importance'] / df['importance'].sum()

    importance_sum = df['importance'].cumsum()
    bound = 0
    for i, v in enumerate(importance_sum):
        if v >= importance_percentile:
            bound = i
            break

    df_selected = df[:bound]
    print(f"group: {group_index}, {bound} of {topk_x.shape[1]} factors "
          f"are selected for importance percentile {importance_percentile:.4f}")
    indices = np.array([renames.index(rename) for rename in df[: 3*bound]['renamed_factor']])
    selected_names = list(df[: 3*bound]['original_factor'])
    x_selected = topk_x[:, indices, :]
    # df_train.to_csv(f'df_train_group{group_index}.csv', index=False)
    # df_valid.to_csv(f'df_valid_group{group_index}.csv', index=False)
    # df_test.to_csv(f'df_test_group{group_index}.csv', index=False)

    train_x_selected = x_selected[:a]
    valid1_x_selected = x_selected[a:b]
    valid2_x_selected = x_selected[b:c]
    test_x_selected = x_selected[c:]

    train_x_selected = np.reshape(np.transpose(train_x_selected, (0, 2, 1)), (-1, len(selected_names)))
    valid1_x_selected = np.reshape(np.transpose(valid1_x_selected, (0, 2, 1)), (-1, len(selected_names)))
    valid2_x_selected = np.reshape(np.transpose(valid2_x_selected, (0, 2, 1)), (-1, len(selected_names)))
    test_x_selected = np.reshape(np.transpose(test_x_selected, (0, 2, 1)), (-1, len(selected_names)))

    pred = model.predict(train_topk_x)
    df_train = pd.DataFrame({'pred': pred, 'y': train_y}, index=pd.MultiIndex.from_product([dates[:a], symbols]))
    # df_train = pd.DataFrame(data=train_x_selected, columns=selected_names,
    #                        index=pd.MultiIndex.from_product([dates[:a], symbols]))
    df_train.index.names = ['date', 'symbol']
    # df_train['pred'] = pred
    # df_train['y'] = train_y
    df_train.reset_index(inplace=True)

    # valid & test
    pred = model.predict(valid1_topk_x)
    df_valid1 = pd.DataFrame({'pred': pred, 'y': valid1_y}, index=pd.MultiIndex.from_product([dates[a:b], symbols]))
    # df_valid1 = pd.DataFrame(data=valid1_x_selected, columns=selected_names,
    #                        index=pd.MultiIndex.from_product([dates[a:b], symbols]))
    df_valid1.index.names = ['date', 'symbol']
    # df_valid1['pred'] = pred
    # df_valid1['y'] = valid1_y
    df_valid1.reset_index(inplace=True)

    o = compute_IC(df_valid1, ['pred'], 'y')
    valid1_ic = o.mean().iloc[0]


    pred = model.predict(valid2_topk_x)
    df_valid2 = pd.DataFrame({'pred': pred, 'y': valid2_y}, index=pd.MultiIndex.from_product([dates[b:c], symbols]))
    # df_valid2 = pd.DataFrame(data=valid2_x_selected, columns=selected_names,
    #                        index=pd.MultiIndex.from_product([dates[b:c], symbols]))
    df_valid2.index.names = ['date', 'symbol']
    # df_valid2['pred'] = pred
    # df_valid2['y'] = valid2_y
    df_valid2.reset_index(inplace=True)
    o = compute_IC(df_valid2, ['pred'], 'y')
    valid2_ic = o.mean().iloc[0]

    pred = model.predict(test_topk_x)
    df_test = pd.DataFrame({'pred': pred, 'y': test_y}, index=pd.MultiIndex.from_product([dates[c:], symbols]))
    # df_test = pd.DataFrame(data=test_x_selected, columns=selected_names,
    #                        index=pd.MultiIndex.from_product([dates[c:], symbols]))
    df_test.index.names = ['date', 'symbol']
    # df_test['pred'] = pred
    # df_test['y'] = test_y
    df_test.reset_index(inplace=True)

    o = compute_IC(df_test, ['pred'], 'y')
    test_ic = o.mean().iloc[0]

    df.to_csv(f'df_importance_xxx.csv', index=False)
    # df_train.to_csv(f'df_train_xxx.csv', index=False)
    # df_valid1.to_csv(f'df_valid1_xxx.csv', index=False)
    # df_valid2.to_csv(f'df_valid2_xxx.csv', index=False)
    # df_test.to_csv(f'df_test_xxx.csv', index=False)

    return group_index, x_selected, df_selected['original_factor'].to_list(), valid1_ic, valid2_ic, test_ic


def group_select_eval(n_group_interval=10, train_valid_test_ratios=(0.6, 0.1, 0.1, 0.2), importance_percentile=0.8):
    result_dir = 'logs/depth=(2, 6)/generation0'
    data_dir = 'data'
    # x_original = np.load(f'{data_dir}/x_4d.npy')
    y = np.load(f'{data_dir}/y.npy')

    times = list(np.load(f'{data_dir}/times.npy'))
    dates = list(np.load(f'{data_dir}/dates.npy'))
    symbols = list(np.load(f'{data_dir}/symbols.npy'))

    topk_array_files = []
    topk_factor_name_files = []
    for directory in os.listdir(result_dir):
        if directory.startswith('group'):
            filename = f"{result_dir}/{directory}/topk_array.npy"
            if os.path.exists(filename):
                topk_array_files.append(filename)
                topk_factor_name_files.append(f"{result_dir}/{directory}/topk_factor_names.txt")
    ray.init(num_gpus=4, num_cpus=8*4, ignore_reinit_error=True)

    print(f"{len(topk_array_files)} groups in total.")

    group_x_list = []
    group_names_list = []
    task_ids = []
    results_dict = {}
    for group_idx, idx in enumerate(range(0, len(topk_array_files), n_group_interval)):
        array_files = topk_array_files[:idx + n_group_interval]
        factor_name_files = topk_factor_name_files[:idx + n_group_interval]

        topk_x = np.concatenate([np.load(f"{file}") for file in array_files], axis=1)

        full_factor_names = []
        for file in factor_name_files:
            with open(file, 'r') as f:
                lines = f.readlines()
                names = []
                for l in lines:
                    idx = l.index(',')
                    a, b = l[:idx], l[idx+1:]
                    names.append(b.strip() + '_' + a.split('_')[1])
            full_factor_names += names

        task_id = eval_group_topk.options(
            num_gpus=1, num_cpus=8,
        ).remote(
            group_idx, topk_x, y, dates,
            symbols, x_names=full_factor_names,
            train_valid12_test_ratios=train_valid_test_ratios,
            importance_percentile=importance_percentile)

        task_ids.append(task_id)

        # if len(task_ids) >= 4:
        #     dones, task_ids = ray.wait(task_ids, num_returns=1)
        #     group_idx, group_x, group_names, valid_ic, test_ic = ray.get(dones)
        #     group_x_list.append(group_x)
        #     group_names_list.extend(group_names)

        #     print(f'group: {group_idx}, valid_ic: {valid_ic:.4f}, test_ic: {test_ic:.4f}')

    if len(task_ids) > 0:
        res_list = ray.get(task_ids)

        for res in res_list:
            group_idx, group_x, group_names, valid1_ic, valid2_ic, test_ic = res
            group_x_list.append(group_x)
            group_names_list.extend(group_names)

            results_dict[str(group_idx)] = [len(group_names), valid1_ic, valid2_ic, test_ic]
            print(f'group: {group_idx}, {len(group_names)} factors selected, valid1_ic: {valid1_ic:.4f}, valid2_ic: {valid2_ic:.4f}, test_ic: {test_ic:.4f}')

    pickle.dump(results_dict, open(f'results_summary_{importance_percentile:.1f}.pkl', 'wb'))
    #     final_x = np.concatenate(group_x_list, axis=1)
    #     final_x_names = group_names_list



@hydra.main(config_path='.', config_name='autofe')
def run(args: DictConfig) -> None:
    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    args.expression_depth = eval(args.expression_depth)
    args.train_valid12_test_ratios = eval(args.train_valid12_test_ratios)
    generation_dir = "generation0"
    expressions_dir = f"{generation_dir}/expressions{args.n_expressions}"
    if not os.path.exists(expressions_dir):
        os.makedirs(expressions_dir)

    random.seed(args.seed)
    ray.init(num_gpus=args.init_num_gpus, num_cpus=args.init_num_cpus, ignore_reinit_error=True)

    input_factors = list(np.load(f'{data_dir}/factors.npy'))
    renamed_input_factors = [f"x{idx}" for idx, _ in enumerate(input_factors)]

    logger.info(f"expression_depth {args.expression_depth}")

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        x = torch.FloatTensor(np.load(f'{data_dir}/x_4d.npy'))
        y = torch.FloatTensor(np.load(f'{data_dir}/y.npy'))
    times = list(np.load(f'{data_dir}/times.npy'))
    dates = list(np.load(f'{data_dir}/dates.npy'))
    symbols = list(np.load(f'{data_dir}/symbols.npy'))

    x_id = ray.put(x)
    y_id = ray.put(y)
    dates_id = ray.put(dates)
    times_id = ray.put(times)
    symbols_id = ray.put(symbols)

    init_population(input_factors=renamed_input_factors, expression_depth=args.expression_depth,
                    n_expressions=args.n_expressions, save_dir=expressions_dir)

    factors = [pickle.load(open(f'{expressions_dir}/{file}', 'rb'))
               for file in os.listdir(f'{expressions_dir}')
               if file.endswith('.pkl') and file.startswith('factor')]

    assert len(factors) == args.n_expressions

    logger.info('=====> Group Evaluation Starting')

    pool = ActorPool([GroupPipeline.options(
        num_gpus=args.pipeline_num_gpus,
        num_cpus=args.pipeline_num_cpus
    ).remote(
        x=x_id, y=y_id, dates=dates_id,
        times=times_id, symbols=symbols_id,
        x_names=renamed_input_factors, args=args)
    for _ in range(args.init_num_gpus)])

    pool.submit(lambda a, _: a.forward_baseline.remote(), None)
    _, df_valid1, df_valid2, df_test = pool.get_next()


    valid1_ic = compute_IC(df_valid1, ['pred'], 'y').mean().iloc[0]
    valid2_ic = compute_IC(df_valid2, ['pred'], 'y').mean().iloc[0]
    test_ic = compute_IC(df_test, ['pred'], 'y').mean().iloc[0]
    ic_threshold = valid1_ic * args.threshold_ratio

    logger.info(f"baseline valid1_ic: {valid1_ic:.4f}, valid2_ic: {valid2_ic:.4f}, test_ic: {test_ic:.4f}.")

    for group_idx, i in enumerate(range(0, len(factors), args.group_size)):
        # group_expressions = {idx: factors[idx] for idx in range(i, i+args.group_size)}
        group_expressions = factors[i: i+args.group_size]


        tuple_args = ForwardArgs(group_idx, group_expressions,
                                 ic_threshold, args.topk, generation_dir)
        pool.submit(lambda a, v: a.forward.remote(v), tuple_args)

    while pool.has_next():
        pool.get_next_unordered()
    logger.info('=====> Group Evaluation Done')


if __name__ == '__main__':
    # run()
    # n_groups = 1
    importance_percentile = 0.9
    group_select_eval(
        n_group_interval=10, importance_percentile=importance_percentile)
