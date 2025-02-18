"""
Utilities for generating performance profile plots.
"""

import os
from typing import Callable, Dict, Any, List
from operator import itemgetter
from collections import defaultdict
import math

import numpy as np

from experiment_utils import files

train_metric = "train_loss"
test_metric = "val_score"


def compute_success_ratios(
    experiment_names: List[str],
    exp_list: Dict[str, Any],
    compute_xy_values: Callable,
    problem_key: Callable,
    method_key: Callable,
    filter_result: Callable,
) -> Dict[str, Any]:
    results: Dict[str, Any] = defaultdict(lambda: defaultdict(list))
    best_results = defaultdict(list)

    # organize experiments by method
    for exp in exp_list:
        exp_metrics = None
        for name in experiment_names:
            try:
                exp_metrics = files.load_experiment(
                    exp,
                    results_dir=os.path.join("results", name),
                    load_metrics=True,
                )["metrics"]
            except:
                print("failed to load")
                continue

        if exp_metrics is None or filter_result(exp_metrics):
            continue

        x_value, success = compute_xy_values(exp_metrics, exp)

        results[method_key(exp)][problem_key(exp)].append([x_value, success])

    for mkey in results.keys():
        for pkey in results[mkey].keys():
            # find the best run by minimizing over x-axis
            successes = list(filter(itemgetter(1), results[mkey][pkey]))
            if len(successes) > 0:
                best_results[mkey].append(min(successes, key=itemgetter(0)))
            else:
                best_results[mkey].append(
                    max(results[mkey][pkey], key=itemgetter(0))
                )

    n_problems = max([len(best_results[key]) for key in best_results.keys()])

    for key in best_results.keys():
        ordered = list(sorted(best_results[key], key=itemgetter(0)))
        values = np.array(ordered).T
        cumulative_successes = np.cumsum(values[1]) / n_problems
        best_results[key] = (values[0], cumulative_successes)

    return best_results, n_problems


def compute_obj_success_ratios(
    experiment_names: List[str],
    exp_list: Dict[str, Any],
    compute_xy_values: Callable,
    problem_key: Callable,
    method_key: Callable,
    filter_result: Callable,
    best_obj: dict[Any, list],
) -> Dict[str, Any]:
    results: Dict[str, Any] = defaultdict(lambda: defaultdict(list))
    best_results = defaultdict(list)

    # organize experiments by method
    metrics = []
    # n_fails = 0

    for exp in exp_list:
        exp_metrics = None
        for name in experiment_names:
            # try:
            exp_metrics = files.load_experiment(
                exp,
                results_dir=os.path.join("results", name),
                load_metrics=True,
            )["metrics"]
            # except:
            #     n_fails += 1
            #     continue

        if exp_metrics is None or filter_result(exp_metrics, exp):
            continue

        metrics.append((exp, exp_metrics))

    for exp, exp_metrics in metrics:
        pkey = problem_key(exp)
        x_value, success = compute_xy_values(exp_metrics, exp, best_obj[pkey])
        lr = exp["opt"]["lr"]
        results[method_key(exp)][pkey].append([x_value, success, lr])

    for mkey in results.keys():
        for pkey in results[mkey].keys():
            # find the best run by minimizing over x-axis
            successes = list(filter(itemgetter(1), results[mkey][pkey]))
            if len(successes) > 0:
                best_results[mkey].append(min(successes, key=itemgetter(0)))
            else:
                best_results[mkey].append(
                    max(results[mkey][pkey], key=itemgetter(0))
                )

    n_problems = max([len(best_results[key]) for key in best_results.keys()])

    for key in best_results.keys():
        ordered = list(sorted(best_results[key], key=itemgetter(0)))
        values = np.array(ordered).T
        cumulative_successes = np.cumsum(values[1]) / n_problems
        best_results[key] = (values[0], cumulative_successes)

    return best_results, n_problems


def compute_acc_success_ratios(
    experiment_names: List[str],
    exp_list: Dict[str, Any],
    compute_xy_values: Callable,
    problem_key: Callable,
    method_key: Callable,
    filter_result: Callable,
) -> Dict[str, Any]:
    results: Dict[str, Any] = defaultdict(lambda: defaultdict(list))
    best_results = defaultdict(list)
    best_acc = {}

    # organize experiments by method
    metrics = []
    for exp in exp_list:
        exp_metrics = None
        for name in experiment_names:
            try:
                exp_metrics = files.load_experiment(
                    exp,
                    results_dir=os.path.join("results", name),
                    load_metrics=True,
                )["metrics"]
            except:
                continue

        if exp_metrics is None or filter_result(exp_metrics, exp):
            continue

        metrics.append((exp, exp_metrics))
        key = problem_key(exp)

        max_acc = np.max(exp_metrics[test_metric])

        if not math.isnan(max_acc) and (
            key not in best_acc or max_acc > best_acc[key]
        ):
            best_acc[key] = max_acc

    for exp, exp_metrics in metrics:
        pkey = problem_key(exp)
        x_value, success = compute_xy_values(exp_metrics, exp, best_acc[pkey])
        results[method_key(exp)][pkey].append([x_value, success])

    for mkey in results.keys():
        for pkey in results[mkey].keys():
            # find the best run by minimizing over x-axis
            successes = list(filter(itemgetter(1), results[mkey][pkey]))
            if len(successes) > 0:
                best_results[mkey].append(min(successes, key=itemgetter(0)))
            else:
                best_results[mkey].append(
                    max(results[mkey][pkey], key=itemgetter(0))
                )

    n_problems = max([len(best_results[key]) for key in best_results.keys()])

    for key in best_results.keys():
        ordered = list(sorted(best_results[key], key=itemgetter(0)))
        values = np.array(ordered).T
        cumulative_successes = np.cumsum(values[1]) / n_problems
        best_results[key] = (values[0], cumulative_successes)

    return best_results, n_problems


def compute_reg_success_ratios(
    experiment_names: List[str],
    exp_list: Dict[str, Any],
    compute_xy_values: Callable,
    problem_key: Callable,
    method_key: Callable,
    filter_result: Callable,
    best_obj: dict[Any, list],
) -> Dict[str, Any]:
    results: Dict[str, Any] = defaultdict(lambda: defaultdict(list))
    best_results = defaultdict(list)

    # organize experiments by method
    metrics = []
    for exp in exp_list:
        exp_metrics = None
        for name in experiment_names:
            try:
                exp_metrics = files.load_experiment(
                    exp,
                    results_dir=os.path.join("results", name),
                    load_metrics=True,
                )["metrics"]
            except:
                continue

        if exp_metrics is None or filter_result(exp_metrics, exp):
            continue

        metrics.append((exp, exp_metrics))

    for exp, exp_metrics in metrics:
        pkey = problem_key(exp)
        x_value, success = compute_xy_values(exp_metrics, exp, best_obj[pkey])
        results[method_key(exp)][pkey].append([x_value, success, exp["reg"]])

    for mkey in results.keys():
        for pkey in results[mkey].keys():
            # find the best run by minimizing over x-axis
            successes = list(filter(itemgetter(1), results[mkey][pkey]))
            if len(successes) > 0:
                best_results[mkey].append(min(successes, key=itemgetter(0)))
            else:
                best_results[mkey].append(
                    max(results[mkey][pkey], key=itemgetter(0))
                )

    n_problems = 20
    final_results = {}
    for mkey in best_results.keys():
        success_by_reg = defaultdict(lambda: 0)
        for _, success, reg in best_results[mkey]:
            success_by_reg[reg] += success

        ordered = list(sorted(success_by_reg.items(), key=itemgetter(0)))
        values = np.array(ordered).T
        total_successes = values[1] / n_problems
        final_results[mkey] = (values[0], total_successes)

    return final_results, n_problems


def compute_step_times(
    experiment_names: List[str],
    exp_list: Dict[str, Any],
    compute_xy_values: Callable,
    problem_key: Callable,
    method_key: Callable,
    filter_result: Callable,
    best_obj: dict[Any, list],
) -> Dict[str, Any]:
    results: Dict[str, Any] = defaultdict(lambda: defaultdict(list))
    best_results = defaultdict(list)
    total_times = defaultdict(list)

    # organize experiments by method
    metrics = []

    for exp in exp_list:
        exp_metrics = None
        for name in experiment_names:
            try:
                exp_metrics = files.load_experiment(
                    exp,
                    results_dir=os.path.join("results", name),
                    load_metrics=True,
                )["metrics"]
            except:
                continue

        if exp_metrics is None or filter_result(exp_metrics, exp):
            continue

        metrics.append((exp, exp_metrics))
        key = problem_key(exp)
        key_short = (key[0], key[1])

        total_times[key_short] += [np.sum(exp_metrics["train_epoch_time"])]

    best_time = {}
    for key in total_times.keys():
        best_time[key] = np.min(total_times[key])

    for exp, exp_metrics in metrics:
        pkey = problem_key(exp)
        pkey_short = (pkey[0], pkey[1])

        x_value, success = compute_xy_values(
            exp_metrics, exp, best_obj[pkey_short], best_time[pkey_short]
        )
        results[method_key(exp)][pkey].append([x_value, success, pkey[2]])

    for mkey in results.keys():
        for pkey in results[mkey].keys():
            # find the best run by minimizing over x-axis
            successes = list(filter(itemgetter(1), results[mkey][pkey]))
            if len(successes) > 0:
                best_results[mkey].append(min(successes, key=itemgetter(0)))
            else:
                best_results[mkey].append(
                    max(results[mkey][pkey], key=itemgetter(0))
                )

    final_results = {}
    for mkey in best_results.keys():
        times_by_steps = defaultdict(list)
        for times, success, steps in best_results[mkey]:
            times_by_steps[steps].append(times)

        ordered = list(sorted(times_by_steps.items(), key=itemgetter(0)))
        keys = [k for k, v in ordered]
        values = np.array([v for k, v in ordered])

        mean = np.median(values, axis=-1)
        upper = np.quantile(values, axis=-1, q=0.75)
        lower = np.quantile(values, axis=-1, q=0.25)

        final_results[mkey] = (keys, mean, lower, upper)

    return final_results
