"""Plot configuration for SLS datasets."""

from copy import deepcopy
from collections.abc import Callable
from typing import Any
from itertools import product

from experiment_utils import utils
from experiment_utils.plotting import defaults

# plot configuration #

# CONSTANTS #

row_key = "batch_size"

repeat_key = "repeat"


figure_labels = {
    "x_labels": {
        "train_loss": "Iterations",
        "val_score": "Iterations",
        "grad_norm": "Iterations",
    },
    "y_labels": {
        128: "Stochastic",
        "full_batch": "Deterministic",
    },
    "col_titles": {
        "train_loss": "Training Objective",
        "val_score": "Test Accuracy",
        "grad_norm": "Gradient Norm",
    },
    "row_titles": {},
}


def line_key(exp_dict):
    """Load line key."""
    method = exp_dict["opt"]
    teleport = exp_dict["init"]["name"]
    name = method["name"]
    key = f"{name}_{teleport}"

    if method.get("momentum", 0) != 0:
        key = key + "_mom"

    return key


line_width = 3
marker_size = 10
markevery = 0.2

settings = defaults.DEFAULT_SETTINGS

settings["y_labels"] = "left_col"
settings["x_labels"] = "bottom_row"
settings["show_legend"] = True
settings["fig_width"] = 6
settings["fig_height"] = 6
settings["legend_fs"] = 27
settings["tick_fs"] = 26
settings["axis_labels_fs"] = 32
settings["subtitle_fs"] = 38
settings["bottom_margin"] = 0.36
settings["legend_cols"] = 3


line_kwargs = {
    "sgd-m_teleport": {
        "c": defaults.line_colors[4],
        "label": f"SGD",
        "linewidth": line_width,
        "marker": defaults.marker_styles[4],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
    "sgd-m_Default": {
        "c": defaults.line_colors[4],
        "label": f"",
        "linewidth": line_width,
        "marker": defaults.marker_styles[4],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "dashed",
    },
    "sgd-m_teleport_mom": {
        "c": defaults.line_colors[3],
        "label": f"SGD-M",
        "linewidth": line_width,
        "marker": defaults.marker_styles[3],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
    "sgd-m_Default_mom": {
        "c": defaults.line_colors[3],
        "label": f"",
        "linewidth": line_width,
        "marker": defaults.marker_styles[3],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "dashed",
    },
    "normalized-sgd_teleport": {
        "c": defaults.line_colors[1],
        "label": f"Norm. SGD",
        "linewidth": line_width,
        "marker": defaults.marker_styles[1],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
    "normalized-sgd_Default": {
        "c": defaults.line_colors[1],
        "label": f"",
        "linewidth": line_width,
        "marker": defaults.marker_styles[1],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "dashed",
    },
    "sps_teleport": {
        "c": defaults.line_colors[2],
        "label": f"SPS",
        "linewidth": line_width,
        "marker": defaults.marker_styles[2],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
    "sps_Default": {
        "c": defaults.line_colors[2],
        "label": f"",
        "linewidth": line_width,
        "marker": defaults.marker_styles[2],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "dashed",
    },
    "gd_ls_teleport": {
        "c": defaults.line_colors[5],
        "label": f"SGD-LS",
        "linewidth": line_width,
        "marker": defaults.marker_styles[5],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
    "gd_ls_Default": {
        "c": defaults.line_colors[5],
        "label": f"",
        "linewidth": line_width,
        "marker": defaults.marker_styles[5],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "dashed",
    },
}


def variation_key(exp_config):
    opt = exp_config["opt"]
    lr = opt.get("lr", 0)
    alpha = opt.get("alpha", 0)

    return (lr, alpha)


metrics = [
    "train_loss",
    "val_score",
    "grad_norm",
]

keep = [
    (("reg"), [0.01]),
    (("batch_size"), ["full_batch"]),
    (("model"), ["smooth_mlp"]),
]

remove: Any = []

limits = {}
ticks = {"val_score": (None, [0.1, 0.3, 0.5, 0.7, 0.9])}
processing_fns = [utils.drop_start(1, lambda key: key[1] == "grad_norm")]
figure_labels = deepcopy(figure_labels)

settings = deepcopy(settings)
settings["y_labels"] = None
settings["bottom_margin"] = 0.32
settings["legend_cols"] = 5

settings["fig_width"] = 6
settings["fig_height"] = 5
settings["legend_fs"] = 24
settings["tick_fs"] = 24
settings["axis_labels_fs"] = 28
settings["subtitle_fs"] = 32


def filter_fn(exp_config):
    opt = exp_config["opt"]

    if opt["name"] != "sgd-m" or opt["momentum"] != 0.9:
        return True

    if exp_config["model"] != "smooth_mlp":
        return True

    if opt["lr"] == 1:
        return False

    return True


mnist_paper_plot_template = {
    "name": "",
    "row_key": row_key,
    "metrics": metrics,
    "line_key": line_key,
    "repeat_key": repeat_key,
    "variation_key": variation_key,
    "target_metric": "train_loss",
    "maximize_target": False,
    "metrics_fn": utils.quantile_metrics,
    "keep": keep,
    "remove": remove,
    "filter_fn": filter_fn,
    "processing_fns": processing_fns,
    "figure_labels": figure_labels,
    "line_kwargs": line_kwargs,
    "log_scale": {
        "train_loss": "log-linear",
        "val_score": "linear-linear",
        "grad_norm": "log-linear",
    },
    "limits": limits,
    "ticks": ticks,
    "settings": settings,
    # "silent_fail": True,
    # "x_key": "iter",
}

deterministic_paper_plot = deepcopy(mnist_paper_plot_template)
deterministic_limits = {
    ("train_loss"): (None, [None, 3]),
    ("val_score"): (None, None),
    ("grad_norm"): (None, None),
}
deterministic_paper_plot["limits"] = deterministic_limits
deterministic_paper_plot["name"] = "figure_5"


stochastic_paper_plot = deepcopy(mnist_paper_plot_template)

stochastic_limits = {
    ("train_loss"): (None, [0.18, 1]),
    ("val_score"): (None, [0.75, 0.975]),
    ("grad_norm"): (None, None),
}


stochastic_paper_plot["keep"] = [
    (("reg"), [0.01]),
    (("batch_size"), [128]),
    (("model"), ["smooth_mlp"]),
]
stochastic_paper_plot["name"] = "figure_13"
stochastic_paper_plot["limits"] = stochastic_limits
stochastic_paper_plot["ticks"] = {}
mnist_paper_plots = [deterministic_paper_plot, stochastic_paper_plot]

keep = [
    (("reg"), [0.01]),
    (("batch_size"), ["full_batch"]),
    (("model"), ["mlp"]),
]

deterministic_non_smooth = deepcopy(deterministic_paper_plot)
deterministic_non_smooth["keep"] = keep
deterministic_non_smooth["name"] = "figure_14"

keep = [
    (("reg"), [0.01]),
    (("batch_size"), [128]),
    (("model"), ["mlp"]),
]

mnist_paper_plots_non_smooth = [deterministic_non_smooth]

PLOTS = {
    "figures_5_13": mnist_paper_plots,
    "figure_14": mnist_paper_plots_non_smooth,
}
