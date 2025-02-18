"""
Plot grid of model fits for 1-D data.
"""

import os
import pickle as pkl
from typing import List, Dict
from functools import reduce
from operator import itemgetter
from collections import defaultdict

import torch
import numpy as np

from experiment_utils import configs, utils, files, command_line
from experiment_utils.plotting import plot_grid, plot_cell, defaults

# merge experiment dictionaries.
from exp_configs import EXPERIMENTS
from plot_configs import PLOTS


if __name__ == "__main__":

    exp_ids = ["figures_6_15_16"]
    plot_names = ["figure_6"]
    figures_dir = "./figures"
    results_dir_base = "./results"
    verbose = False
    debug = False
    log_file = None

    logger_name = reduce(lambda acc, x: f"{acc}{x}_", exp_ids, "")
    logger = utils.get_logger(logger_name, verbose, debug, log_file)

    # lookup experiment #
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            raise ValueError(f"Experiment id {exp_id} is not in the experiment list!")

    config_list: List[Dict] = reduce(
        lambda acc, eid: acc + EXPERIMENTS[eid], exp_ids, []
    )
    results_dir = [os.path.join(results_dir_base, eid) for eid in exp_ids]

    logger.info(f"\n\n====== Making {plot_names} from {exp_ids} results ======\n")

    for plot_name in plot_names:
        dest = os.path.join(figures_dir, "_".join(exp_ids))
        plot_config_list = PLOTS[plot_name]

        for i, plot_config in enumerate(plot_config_list):
            logger.info(f"\n\n Creating {i+1}/{len(plot_config_list)} \n")

            exp_list = configs.expand_config_list(config_list)
            if len(exp_list) == 0:
                exp_list = config_list

            exp_list = configs.filter_dict_list(
                exp_list,
                keep=plot_config["keep"],
                remove=plot_config["remove"],
                filter_fn=plot_config["filter_fn"],
            )

            exp_grid = configs.make_grid(
                exp_list=exp_list,
                row_key=plot_config["row_key"],
                col_key=plot_config["col_key"],
                line_key=plot_config["line_key"],
                repeat_key=plot_config["repeat_key"],
                variation_key=plot_config["variation_key"],
            )

            target_metric = plot_config.get("target_metric", None)

            def call_fn(repeat_dict: dict, keys):
                target_values = []
                metrics = {}
                for key, config in repeat_dict.items():
                    results = files.load_experiment(
                        config,
                        load_metrics=True,
                        load_model=True,
                        results_dir=results_dir,
                    )
                    train_loss = results["metrics"]["train_loss"]
                    results["metrics"]["train_loss"] = np.minimum.accumulate(train_loss)
                    metrics[key] = results["metrics"]

                    value = results["metrics"][target_metric][-1]
                    if np.isnan(value):
                        value = np.inf
                    target_values.append((key, value))

                best_key = min(target_values, key=itemgetter(1))[0]
                best_metrics = metrics[best_key][target_metric]

                return best_metrics

            metric_grid = configs.call_on_grid(exp_grid, call_fn)

            metric_grid = files.compute_metrics(
                metric_grid,
                metric_fn=plot_config["metrics_fn"],
                x_key=plot_config.get("x_key", None),
                x_vals=plot_config.get("x_vals", None),
            )

            plot_grid.plot_grid(
                plot_fn=plot_config.get("plot_fn", plot_cell.make_convergence_plot),
                results=metric_grid,
                figure_labels=plot_config["figure_labels"],
                line_kwargs=plot_config["line_kwargs"],
                limits=plot_config["limits"],
                log_scale=plot_config["log_scale"],
                base_dir=os.path.join(
                    dest, f"{plot_name}", f"{plot_config['name']}.pdf"
                ),
                settings=plot_config["settings"],
            )

    logger.info("Plotting done.")
