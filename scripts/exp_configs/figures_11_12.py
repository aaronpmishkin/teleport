from copy import deepcopy
import pickle as pkl
from itertools import product

step_sizes = [1e3, 100, 10, 5, 2, 1, 1e-1, 1e-2, 1e-3, 1e-4]
reg_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

with open("./scripts/exp_configs/uci_constants.pkl", "rb") as f:
    uci_constants = pkl.load(f)

datasets = [
    "blood",
    "breast-cancer",
    "chess-krvkp",
    "congressional-voting",
    "conn-bench-sonar-mines-rocks",
    "credit-approval",
    "cylinder-bands",
    "hill-valley",
    "horse-colic",
    "ilpd-indian-liver",
    "ionosphere",
    "magic",
    "mammographic",
    "musk-1",
    "ozone",
    "pima",
    "tic-tac-toe",
    "titanic",
    "ringnorm",
    "spambase",
]

uci_template = {
    "dataset": None,
    "dataset_kwargs": {
        "test_prop": 0.2,
        "valid_prop": 0.2,
        "use_valid": False,
    },
    "model": ["mlp"],
    "model_kwargs": {
        "output_size": 1,
        "hidden_sizes": [[100, 100], []],
        "bias": True,
    },
    "loss_func": "logistic",
    "reg": None,
    "score_func": "logistic_accuracy",
    "init": {
        "name": "teleport",
        "max_steps": 50,
        "allow_sublevel": True,
        "line_search": True,
        "rho": 0.1,
        "epochs_to_teleport": None,
        "max_steps": [1, 10, 25, 50],
    },
    "opt": [
        {
            "name": "sps",
            "lr": step_sizes,
            "weight_decay": 0,
            "lr_schedule": "constant",
            "prox": False,
        },
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
            "momentum": 0.9,
            "dampening": 0.9,
            "lr_schedule": "constant",
        },
        {
            "name": "sgd-m",
            "lr": step_sizes,
            "weight_decay": 0,
            "momentum": 0,
            "dampening": 0,
            "lr_schedule": "constant",
        },
        {
            "name": "normalized-sgd",
            "lr": step_sizes,
            "weight_decay": 0,
        },
    ],
    "batch_size": None,  # these get set below
    "max_epoch": None,
    "eval_freq": None,
    "repeat": list(range(1)),
}

uci_teleport_long = []

teleport_schedule_det = list(range(5, 1000, 50))
teleport_schedule_stoc = list(range(5, 200, 10))
configurations = [
    (64, 5, 200, teleport_schedule_stoc),
    (64, 5, 200, None),
    ("full_batch", 1, 1000, teleport_schedule_det),
    ("full_batch", 1, 20000, None),
]

for (
    dataset,
    lam,
    (batch_size, eval_freq, epochs, teleport_schedule),
) in product(datasets, reg_values, configurations):
    template_copy = deepcopy(uci_template)

    template_copy["dataset"] = dataset
    template_copy["reg"] = lam

    if teleport_schedule is not None:
        template_copy["init"]["epochs_to_teleport"] = [teleport_schedule]
    else:
        template_copy["init"] = {"name": "Default"}

    template_copy["batch_size"] = batch_size
    template_copy["eval_freq"] = eval_freq
    template_copy["max_epoch"] = epochs
    uci_teleport_long.append(template_copy)

EXPERIMENTS: dict[str, list] = {
    "figures_11_12": uci_teleport_long,
}
