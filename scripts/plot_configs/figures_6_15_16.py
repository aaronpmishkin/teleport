"""Plot configuration for SLS datasets."""

from copy import deepcopy
from collections.abc import Callable
from typing import Any
from itertools import product
import pickle as pkl

from experiment_utils import utils
from experiment_utils.plotting import defaults

# plot configuration #

# CONSTANTS #

repeat_key = "repeat"

with open("scripts/optimal_values.pkl", "rb") as f:
    logreg_best_obj, network_best_obj = pkl.load(f)


processing_fns = [
    utils.drop_start(1, lambda key: key[1] == "grad_norm"),
    utils.cumulative_min(lambda key: key[1] == "train_loss"),
]


remove: Any = []


def line_key(exp_dict):
    """Load line key."""
    method = exp_dict["opt"]
    teleport = exp_dict["init"]["name"]
    key = f"{method['name']}_{teleport}"

    if "momentum" in method and method["momentum"] != 0:
        key += "_mom"

    if "line_search" in method and method["line_search"] != "none":
        key += "_ls"

    return key


line_width = 4
marker_size = 10

line_kwargs = {
    "sgd-m_teleport": {
        "c": defaults.line_colors[4],
        "label": f"GD",
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
        "label": f"GD-M",
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
        "label": f"Norm. GD",
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
        "label": f"GD-LS",
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
    "newton-cg_Default": {
        "c": defaults.line_colors[6],
        "label": f"Newton-CG",
        "linewidth": line_width,
        "marker": defaults.marker_styles[6],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
    "newton-cg_Default_ls": {
        "c": defaults.line_colors[7],
        "label": f"Newton-CG-LS",
        "linewidth": line_width,
        "marker": defaults.marker_styles[8],
        "markevery": 0.2,
        "markersize": marker_size,
        "linestyle": "solid",
    },
}


def variation_key(exp_config):
    lr = exp_config.get("opt").get("lr", 1)
    alpha = exp_config.get("opt").get("alpha", 1)

    return (lr, alpha)


dataset_map = {
    "[100, 100]": [
        "breast-cancer",
        "horse-colic",
        "ozone",
    ],
    "[]": [
        "chess-krvkp",
        "hill-valley",
        "tic-tac-toe",
    ],
}

row_key = "dataset"

metrics = [
    "train_loss",
    "val_score",
    "grad_norm",
]

figure_labels = {
    "x_labels": {
        "train_loss": "Iterations",
        "val_score": "Iterations",
        "grad_norm": "Gradient Norm",
    },
    "y_labels": {
        "breast-cancer": "Breast Cancer",
        "hill-valley": "Hill Valley",
        "chess-krvkp": "Chess",
        "tic-tac-toe": "Tic-Tac-Toe",
        "ozone": "Ozone",
        "horse-colic": "Horse Colic",
    },
    "col_titles": {
        "train_loss": "Training Objective",
        "val_score": "Test Accuracy",
        "grad_norm": "Gradient Norm",
    },
    "row_titles": {},
}


settings = defaults.DEFAULT_SETTINGS

settings = deepcopy(settings)
settings["show_legend"] = True
settings["y_labels"] = "left_col"
settings["x_labels"] = "bottom_row"
settings["bottom_margin"] = 0.45
settings["legend_cols"] = 4

settings["fig_width"] = 6
settings["fig_height"] = 4
settings["legend_fs"] = 27
settings["tick_fs"] = 26
settings["axis_labels_fs"] = 32
settings["subtitle_fs"] = 32
settings["titles_fs"] = 42

limits = {}

newton_appendix_plots = []

for hidden_sizes in [[], [100, 100]]:
    datasets = dataset_map[str(hidden_sizes)]
    keep = [
        (("reg"), [0.0]),
        (("dataset"), datasets),
        (("model_kwargs", "hidden_sizes"), [hidden_sizes]),
        (("model"), ["mlp"]),
    ]
    figure_labels = deepcopy(figure_labels)
    name = "figure_16" if hidden_sizes == [100, 100] else "figure_15"
    newton_appendix_plots += [
        deepcopy(
            {
                "name": name,
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
                "filter_fn": None,
                "processing_fns": processing_fns,
                "figure_labels": figure_labels,
                "line_kwargs": line_kwargs,
                "log_scale": {
                    "train_loss": "log-linear",
                    "val_score": "linear-linear",
                    "grad_norm": "log-linear",
                },
                "limits": limits,
                "settings": settings,
                # "silent_fail": True,
                # "x_key": "iter",
            }
        )
    ]

figure_labels = {
    "x_labels": {
        "[]": "Epochs",
        "[100, 100]": "Epochs",
    },
    "y_labels": {
        "[]": "Convex",
        "[100, 100]": "Non-Convex",
    },
    "col_titles": {
        "breast-cancer": "Breast Cancer",
        "hill-valley": "Hill Valley",
        "chess-krvkp": "Chess",
        "tic-tac-toe": "Tic-Tac-Toe",
        "ozone": "Ozone",
        "horse-colic": "Horse Colic",
    },
    "row_titles": {},
}


def row_key(exp_config):
    return str(exp_config["model_kwargs"]["hidden_sizes"])


def col_key(exp_config):
    return exp_config["dataset"]


def filter_fn(exp_config):
    dataset_name = exp_config["dataset"]
    hidden_sizes = exp_config["model_kwargs"]["hidden_sizes"]
    datasets_to_plot = dataset_map[str(hidden_sizes)]

    return dataset_name in datasets_to_plot


keep = [
    (("reg"), [0.0]),
    (("model"), ["mlp"]),
]
figure_labels = deepcopy(figure_labels)
name = "figure_6"

settings = deepcopy(settings)
settings["col_titles"] = "every_cell"
settings["vspace"] = 0.45
settings["bottom_margin"] = 0.32
settings["fig_height"] = 3.5
settings["subtitle_fs"] = 32

newton_paper_plot = [
    {
        "name": name,
        "row_key": row_key,
        "col_key": col_key,
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
            "[]": "log-linear",
            "[100, 100]": "log-linear",
        },
        "limits": limits,
        "settings": settings,
        # "silent_fail": True,
        # "x_key": "iter",
    }
]

PLOTS = {
    "figures_15_16": newton_appendix_plots,
    "figure_6": newton_paper_plot,
}
