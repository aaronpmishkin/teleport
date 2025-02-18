import os
from functools import partial
from itertools import product
import pickle as pkl

import numpy as np

from experiment_utils.plotting.defaults import DEFAULT_SETTINGS
from experiment_utils import configs

from exp_configs import EXPERIMENTS  # type: ignore
from performance_profile import compute_obj_success_ratios  # type: ignore
from experiment_utils.plotting import defaults

from matplotlib import pyplot as plt  # type: ignore

plt.rcParams.update({"text.usetex": True})

marker_spacing = 0.1
marker_size = 16
line_width = 5
max_x = 1e3
min_x = 1e-2

settings = DEFAULT_SETTINGS
settings["titles_fs"] = 24
settings["axis_labels_fs"] = 22
settings["legend_fs"] = 20
settings["ticks_fs"] = 18
settings["wspace"] = 0.18

marker_size = 8
line_width = 3

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


def method_key(exp_dict):
    """Load line key."""
    method = exp_dict["opt"]
    teleport = exp_dict["init"]["name"]
    key = f"{method['name']}_{teleport}"

    if "momentum" in method and method["momentum"] != 0:
        key += "_mom"

    return key


def problem_key(exp_dict):
    dataset_name = exp_dict["dataset"]
    lam = exp_dict["reg"]

    return (dataset_name, lam)


def filter_result(exp_metrics, exp_config, logreg=True, batch_size=64, rho=0.1):
    """Remove experiments corresponding to null models."""
    if exp_config["init"].get("rho", rho) != rho:
        return True

    if exp_config["batch_size"] != batch_size:
        return True

    if logreg and len(exp_config["model_kwargs"]["hidden_sizes"]) != 0:
        return True

    if not logreg and len(exp_config["model_kwargs"]["hidden_sizes"]) == 0:
        return True

    return False


def compute_xy_values_tol(exp_metrics, exp_config, min_obj, tol):
    obj = exp_metrics["train_loss"]
    start_obj = exp_metrics["train_loss"][0]

    max_obj = np.max(obj)
    if np.isnan(max_obj):
        max_obj = np.inf
    rel_diff = (obj - min_obj) / (start_obj - min_obj)

    thresholded_tols = rel_diff <= tol
    success = np.any(thresholded_tols) and (max_obj < start_obj * 10)

    best_ind = np.argmax(thresholded_tols)

    time = len(exp_metrics["train_loss"])
    if success:
        # time = np.sum(time_list[0 : best_ind + 1])
        # use iterations instead of time.
        time = best_ind + 1

    return time, success


exp_list = configs.expand_config_list(EXPERIMENTS["figures_4_8_17_18"])

tol_list = [0.2, 0.1, 0.05, 0.03, 0.01]
batch_size_list = [64, "full_batch"]

limits = {
    64: {
        "logreg": 1e-3,
        "nn": 1e-3,
    },
    "full_batch": {
        "logreg": 1e-2,
        "nn": 1e-3,
    },
}

problem_to_tol = {
    (64, True): 0.05,
    ("full_batch", True): 0.15,
    (64, False): 0.05,
    ("full_batch", False): 0.1,
}
with open("scripts/optimal_values.pkl", "rb") as f:
    logreg_best_obj, network_best_obj = pkl.load(f)

rho = 0.1

for (logreg,) in product([True, False]):
    if logreg:
        best_obj = logreg_best_obj
    else:
        best_obj = network_best_obj

    tol = problem_to_tol[(64, logreg)]
    compute_xy_values = partial(compute_xy_values_tol, tol=tol)
    print("Starting stochastic, logreg:", logreg)

    (
        stochastic_success_ratios,
        stochastic_n_problems,
    ) = compute_obj_success_ratios(
        ["figures_4_8_17_18"],
        exp_list,
        compute_xy_values,
        problem_key,
        method_key,
        partial(filter_result, logreg=logreg, batch_size=64, rho=rho),
        best_obj,
    )

    print("Starting deterministic, logreg:", logreg)
    tol = problem_to_tol[("full_batch", logreg)]
    compute_xy_values = partial(compute_xy_values_tol, tol=tol)
    (
        deterministic_success_ratios,
        deterministic_n_problem,
    ) = compute_obj_success_ratios(
        ["figures_4_8_17_18"],
        exp_list,
        compute_xy_values,
        problem_key,
        method_key,
        partial(filter_result, logreg=logreg, batch_size="full_batch", rho=rho),
        best_obj,
    )

    fig = plt.figure(figsize=(12, 4.8))
    spec = fig.add_gridspec(ncols=2, nrows=1)
    ax0 = fig.add_subplot(spec[0, 0])

    # ax0.set_ylim(0, 0.8)
    ax0.set_xscale("log")

    n_vals = 15

    final_times = []
    first_times = []

    for key, (x, y) in stochastic_success_ratios.items():
        first_times.append(x[0])
        final_times.append(x[-1])

    min_x = np.min(first_times)
    max_x = np.max(final_times)
    ax0.set_xlim(min_x, max_x)

    def extend_data(x, y):
        last_val = y[-1]
        x = np.concatenate([x, np.linspace(x[-1], max_x, n_vals)])
        y = np.concatenate([y, np.repeat(last_val, n_vals)])
        return x, y

    for key, (x, y) in stochastic_success_ratios.items():
        x, y = extend_data(np.squeeze(x), np.squeeze(y))
        ax0.plot(x, y, alpha=settings["line_alpha"], **line_kwargs[key])

    ax0.axhline(y=0.50, linestyle="--", linewidth="4", c="k")
    ax0.set_title("Stochastic", fontsize=settings["subtitle_fs"])
    ax0.set_ylabel("Prop. of Problems Solved", fontsize=settings["axis_labels_fs"])
    ax0.set_xlabel("Iterations", fontsize=settings["axis_labels_fs"])
    ax0.tick_params(labelsize=settings["tick_fs"])

    ax1 = fig.add_subplot(spec[0, 1])

    # ax1.set_ylim(0, 0.8)
    ax1.set_xscale("log")

    final_times = []
    first_times = []
    for key, (x, y) in deterministic_success_ratios.items():
        first_times.append(x[0])
        final_times.append(x[-1])

    min_x = np.min(first_times)
    max_x = np.max(final_times)
    ax1.set_xlim(min_x, max_x)

    for key, (x, y) in deterministic_success_ratios.items():
        x, y = extend_data(np.squeeze(x), np.squeeze(y))
        ax1.plot(x, y, alpha=settings["line_alpha"], **line_kwargs[key])

    ax1.axhline(y=0.50, linestyle="--", linewidth="4", c="k")
    ax1.set_title("Deterministic", fontsize=settings["subtitle_fs"])
    ax1.set_xlabel("Iterations", fontsize=settings["axis_labels_fs"])
    ax1.tick_params(labelsize=settings["tick_fs"])

    handles0, labels0 = ax0.get_legend_handles_labels()
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles, labels = handles0 + handles1, labels0 + labels1
    # handles, labels = handles0, labels0

    handles_to_plot, labels_to_plot = [], []
    for i, label in enumerate(labels):
        if label not in labels_to_plot:
            handles_to_plot.append(handles[i])
            labels_to_plot.append(label)

    legend = fig.legend(
        handles=handles_to_plot,
        labels=labels_to_plot,
        loc="lower center",
        borderaxespad=0.1,
        fancybox=False,
        shadow=False,
        ncol=len(handles_to_plot),
        fontsize=settings["legend_fs"],
        frameon=False,
    )

    plt.tight_layout()
    fig.subplots_adjust(
        wspace=settings["wspace"],
        hspace=settings["vspace"],
        bottom=0.25,
    )

    if logreg:
        figure_name = f"figure_8"
    else:
        figure_name = f"figure_4"

    os.makedirs("figures/", exist_ok=True)
    plt.savefig(f"figures/{figure_name}.pdf")

    plt.close()
