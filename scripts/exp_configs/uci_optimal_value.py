from copy import deepcopy
import pickle as pkl
from itertools import product

step_sizes = [1e3, 1e2, 1e1, 5, 2, 1, 1e-1, 1e-2, 1e-3, 1e-4]
reg_values = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

with open("./scripts/exp_configs/uci_constants.pkl", "rb") as f:
    uci_constants = pkl.load(f)

# try half of the datasets to start
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
    },
    "opt": [
        {
            "name": "sgd-m",
            "lr": step_sizes,
            "weight_decay": 0,
            "momentum": 0.9,
            "dampening": 0.9,
            "lr_schedule": "constant",
        },
    ],
    "batch_size": None,  # these get set below
    "max_epoch": None,
    "eval_freq": None,
    "repeat": list(range(1)),
}

uci_teleport_long = []

teleport_schedule_stoc = list(range(5, 500, 10))
teleport_schedule_det = list(range(5, 2500, 50))
configurations = [
    (64, 5, 500, teleport_schedule_stoc),
    ("full_batch", 1, 2500, teleport_schedule_det),
]

for (
    dataset,
    lam,
    (batch_size, eval_freq, epochs, teleport_schedule),
) in product(datasets, reg_values, configurations):
    template_copy = deepcopy(uci_template)

    template_copy["init"]["epochs_to_teleport"] = [teleport_schedule]
    template_copy["dataset"] = dataset
    template_copy["reg"] = lam
    template_copy["batch_size"] = batch_size
    template_copy["eval_freq"] = eval_freq
    template_copy["max_epoch"] = epochs
    uci_teleport_long.append(template_copy)

EXPERIMENTS: dict[str, list] = {
    "uci_optimal_value": uci_teleport_long,
}
