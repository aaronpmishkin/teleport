"""Plot configuration for SLS datasets."""

from copy import deepcopy
from collections.abc import Callable
from typing import Any
from itertools import product

from experiment_utils import utils
from experiment_utils.plotting import defaults

# plot configuration #

# CONSTANTS #

row_key = "dataset"
line_key = ("init", "alpha")

repeat_key = "repeat"

combined_metrics = [
    "tele_objective",
    "tele_kkt_gap",
    "tele_constr_violation",
]

metrics_1 = [
    "tele_objective",
]

metrics_2 = [
    "tele_kkt_gap",
]

metrics_3 = [
    "tele_constr_violation",
]

processing_fns: list[Callable] = []

combined_figure_labels = {
    "x_labels": {
        "tele_objective": "Iterations",
        "tele_kkt_gap": "Iterations",
        "tele_constr_violation": "Iterations",
    },
    "y_labels": {},
    "col_titles": {
        "tele_objective": "Gradient Norm",
        "tele_kkt_gap": "KKT Residual",
        "tele_constr_violation": "Constraint Gap",
        "tele_time": "Time (s)",
    },
    "row_titles": {},
}

figure_labels_1 = {
    "x_labels": {
        "tele_objective": "",
        "tele_kkt_gap": "",
        "tele_constr_violation": "",
    },
    "y_labels": {},
    "col_titles": {
        "tele_objective": "Gradient Norm",
        "tele_kkt_gap": "KKT Residual",
        "tele_constr_violation": "Constraint Gap",
        "tele_time": "Time (s)",
    },
    "row_titles": {},
}

figure_labels_3 = {
    "x_labels": {
        "tele_objective": "Iterations",
        "tele_kkt_gap": "Iterations",
        "tele_constr_violation": "Iterations",
        "tele_time": "Iterations",
    },
    "y_labels": {},
    "col_titles": {
        "tele_objective": "Gradient Norm",
        "tele_kkt_gap": "KKT Residual",
        "tele_constr_violation": "Constraint Gap",
        "tele_time": "Time (s)",
    },
    "row_titles": {},
}

limits = {
    "tele_objective": ([-5, 250], None),
    "tele_kkt_gap": ([-5, 250], None),
    "tele_constr_violation": ([-5, 250], None),
    "tele_time": ([-5, 250], None),
}

settings = defaults.DEFAULT_SETTINGS
settings["y_labels"] = "left_col"
settings["x_labels"] = "bottom_row"
settings["legend_cols"] = 3
settings["show_legend"] = False

settings["bottom_margin"] = 0.18
settings["legend_cols"] = 3

settings["fig_width"] = 6
settings["fig_height"] = 1.5
settings["legend_fs"] = 27
settings["tick_fs"] = 14
settings["axis_labels_fs"] = 16
settings["subtitle_fs"] = 18


remove: Any = []
keep: Any = []

line_kwargs = None

variation_key = ("opt", "lr")

base_config = {
    "name": "teleport_metrics",
    "row_key": row_key,
    "metrics": None,
    "line_key": line_key,
    "repeat_key": repeat_key,
    "variation_key": variation_key,
    "target_metric": "tele_objective",
    "maximize_target": False,
    "metrics_fn": utils.quantile_metrics,
    "keep": keep,
    "remove": remove,
    "filter_fn": None,
    "processing_fns": processing_fns,
    "figure_labels": None,
    "line_kwargs": line_kwargs,
    "log_scale": {
        "tele_objective": "log-linear",
        "tele_kkt_gap": "log-linear",
        "tele_constr_violation": "log-linear",
    },
    # "limits": limits[reg][dataset],
    "limits": limits,
    "settings": settings,
    "silent_fail": True,
    # "x_key": "tele_time"
}

conf_1 = deepcopy(base_config)
conf_1["name"] = "figure_3_top"
conf_1["metrics"] = metrics_1
conf_1["figure_labels"] = figure_labels_1

conf_2 = deepcopy(base_config)
conf_2["name"] = "figure_3_middle"
conf_2["metrics"] = metrics_2
conf_2["figure_labels"] = figure_labels_1
conf_2["target_metric"] = "tele_kkt_gap"

conf_3 = deepcopy(base_config)
conf_3["settings"]["bottom_margin"] = 0.31
conf_3["settings"]["fig_height"] = 1.7
conf_3["name"] = "figure_3_bottom"
conf_3["metrics"] = metrics_3
conf_3["figure_labels"] = figure_labels_3
conf_3["target_metric"] = "tele_constr_violation"

plot_configs = [conf_1, conf_2, conf_3]

PLOTS = {
    "figure_3": plot_configs,
}
