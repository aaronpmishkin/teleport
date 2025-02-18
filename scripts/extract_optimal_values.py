from copy import deepcopy
from collections import defaultdict
import pickle as pkl
from itertools import product
import math
import os

import numpy as np

from experiment_utils import files
from experiment_utils import configs

from exp_configs import EXPERIMENTS  # type: ignore


exp_list = configs.expand_config_list(EXPERIMENTS["uci_optimal_value"])


def problem_key(exp_dict):
    dataset_name = exp_dict["dataset"]
    lam = exp_dict["reg"]

    return (dataset_name, lam)


logreg_best_obj = {}
network_best_obj = {}

logreg_objectives = defaultdict(list)
network_objectives = defaultdict(list)

for exp in exp_list:
    exp_metrics = None
    exp_metrics = files.load_experiment(
        exp,
        results_dir=os.path.join("results", "uci_optimal_value"),
        load_metrics=True,
    )["metrics"]

    key = problem_key(exp)
    logreg = len(exp["model_kwargs"]["hidden_sizes"]) == 0
    train_loss = np.array(exp_metrics["train_loss"])
    nan_indices = np.isnan(exp_metrics["train_loss"])
    train_loss[nan_indices] = np.inf

    if logreg:
        logreg_objectives[key].append(np.min(train_loss))
    else:
        network_objectives[key].append(np.min(train_loss))

for key in logreg_objectives.keys():
    logreg_best_obj[key] = min(logreg_objectives[key])

for key in network_objectives.keys():
    network_best_obj[key] = min(network_objectives[key])

with open("scripts/optimal_values.pkl", "wb") as f:
    pkl.dump((logreg_best_obj, network_best_obj), f)
