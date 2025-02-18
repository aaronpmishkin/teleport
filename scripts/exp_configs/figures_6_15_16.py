from copy import deepcopy
import pickle as pkl
from itertools import product

step_sizes = [1e3, 100, 10, 5, 2, 1, 1e-1, 1e-2, 1e-3, 1e-4]
step_sizes = [1e3, 100, 10, 5, 2, 1, 1e-1, 1e-2, 1e-3, 1e-4]
reg_values = [0.0]
max_sub_iters = [25]


datasets = [
    "breast-cancer",
    "chess-krvkp",
    "hill-valley",
    "horse-colic",
    "ozone",
    "tic-tac-toe",
]

uci_template = {
    "dataset": datasets,
    "dataset_kwargs": {
        "test_prop": 0.2,
        "valid_prop": 0.2,
        "use_valid": False,
    },
    "model": ["mlp"],
    "model_kwargs": {
        "output_size": 1,
        "hidden_sizes": [[], [100, 100]],
        "bias": True,
    },
    "loss_func": "logistic",
    "score_func": "logistic_accuracy",
    "init": [
        {
            "name": "teleport",
            "max_steps": max_sub_iters,
            "max_backtracks": 20,
            "beta": 0.5,
            "allow_sublevel": True,
            "line_search": True,
            "rho": [0.1],
            "epochs_to_teleport": [
                list(range(0, 100, 1)),
            ],
        },
    ],
    "opt": [
        {
            "name": "gd_ls",
            "lr": 1,
            "weight_decay": 0,
            "alpha": [1e-3, 1e-2, 0.1, 0.5],
            "beta": 0.8,
        },
        {
            "name": "sgd-m",
            "lr": step_sizes,
            "weight_decay": 0,
            "momentum": 0,
            "dampening": 0,
            "lr_schedule": "constant",
        },
    ],
    "reg": reg_values,
    "batch_size": "full_batch",
    "max_epoch": 100,
    "eval_freq": 1,
    "repeat": list(range(1)),
}

newton_cg = [
    {
        "name": "newton-cg",
        "lr": 1e-3,
        "cg_max_iter": max_sub_iters,
        "line_search": "strong-wolfe",
        "disp": False,
        "max_iter": 1,
        "xtol": 1e-20,
        "clear_grad": True,
    },
    {
        "name": "newton-cg",
        "lr": step_sizes,
        "cg_max_iter": max_sub_iters,
        "line_search": "none",
        "disp": False,
        "max_iter": 1,
        "xtol": 1e-20,
        "clear_grad": True,
    },
]

newton_cg_exp = deepcopy(uci_template)
newton_cg_exp["opt"] = newton_cg
newton_cg_exp["init"] = {"name": "Default"}


EXPERIMENTS: dict[str, list] = {
    "figures_6_15_16": [uci_template, newton_cg_exp],
}
