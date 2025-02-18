"""Plot configuration for SLS datasets."""

from copy import deepcopy
from collections.abc import Callable
from typing import Any
from itertools import product

from experiment_utils import utils
from experiment_utils.plotting import defaults

# plot configuration #

# CONSTANTS #

repeat_key = "repeat"


def line_key(exp_dict):
    """Load line key."""
    method = exp_dict["opt"]
    teleport = exp_dict["init"]["name"]
    key = f"{method['name']}_{teleport}"

    if "momentum" in method and method["momentum"] != 0:
        key += "_mom"

    return key


remove: Any = []


line_width = 3
marker_size = 10

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
    lr = exp_config.get("opt").get("lr", 1)
    alpha = exp_config.get("opt").get("alpha", 1)

    return (lr, alpha)


dataset_map = {
    "[100, 100]": [
        "credit-approval",
        "pima",
        "ringnorm",
    ],
    "[]": [
        "chess-krvkp",
        "ionosphere",
        "tic-tac-toe",
    ],
}

row_key = "dataset"

metrics = [
    "train_loss",
    "val_score",
    "grad_norm",
]
limits = {
    # ("ilpd-indian-liver", "train_loss"): (None, [0.49, 0.55]),
    # ("ozone", "train_loss"): (None, [0.085, 0.2]),
    # ("spambase", "train_loss"): (None, [0.16, 0.5]),
    # ("ilpd-indian-liver", "val_score"): (None, [0.5, 0.72]),
    # ("ozone", "val_score"): (None, [0.8, 0.98]),
    # ("spambase", "val_score"): (None, [0.75, 0.95]),
    # ("ilpd-indian-liver", "train_loss"): (None, [0.49, 0.55]),
    # ("ozone", "train_loss"): (None, [0.085, 0.2]),
    # ("spambase", "train_loss"): (None, [0.16, 0.5]),
    # ("ilpd-indian-liver", "val_score"): (None, [0.5, 0.72]),
    # ("ozone", "val_score"): (None, [0.8, 0.98]),
    # ("spambase", "val_score"): (None, [0.75, 0.95]),
}

figure_labels = {
    "x_labels": {
        "train_loss": "Iterations",
        "val_score": "Iterations",
    },
    "y_labels": {
        "breast_cancer": "Breast Cancer",
        "pima": "Pima",
        "credit-approval": "Credit Approval",
        "ringnorm": "Ringnorm",
        "chess-krvkp": "Chess",
        "ionosphere": "Ionosphere",
        "tic-tac-toe": "Tic-Tac-Toe",
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
settings["y_labels"] = "left_col"
settings["x_labels"] = "bottom_row"
settings["show_legend"] = True
settings["bottom_margin"] = 0.3
settings["legend_cols"] = 5

settings["fig_width"] = 6
settings["fig_height"] = 6
settings["legend_fs"] = 27
settings["tick_fs"] = 26
settings["axis_labels_fs"] = 32
settings["subtitle_fs"] = 38
processing_fns = [utils.drop_start(1, lambda key: key[1] == "grad_norm")]

uci_paper_plots = []

reg_map = {
    "[]": 1e-6,
    "[100, 100]": 0.01,
}

for hidden_sizes in [[], [100, 100]]:
    reg = reg_map[str(hidden_sizes)]
    datasets = dataset_map[str(hidden_sizes)]
    keep = [
        (("reg"), [reg]),
        (("batch_size"), ["full_batch"]),
        (("dataset"), datasets),
        (("model_kwargs", "hidden_sizes"), [hidden_sizes]),
    ]
    name = "figure_18" if hidden_sizes == [100, 100] else "figure_17"
    uci_paper_plots += [
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

PLOTS = {
    "figures_17_18": uci_paper_plots,
}
