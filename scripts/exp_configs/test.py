from copy import deepcopy

datasets = [
    "blood",
    "breast-cancer",
    "congressional-voting",
    "hill-valley",
    "magic",
    "monks-1",
    "musk-1",
    "ozone",
    "pima",
    "tic-tac-toe",
    "titanic",
    "ringnorm",
]

uci_template = {
    "dataset": datasets[1],
    "dataset_kwargs": {
        "test_prop": 0.2,
        "valid_prop": 0.2,
        "use_valid": False,
    },
    "model": "mlp",
    "model_kwargs": {
        "output_size": 1,
        "hidden_sizes": [[]],
        "bias": True,
    },
    "loss_func": "logistic",
    "reg": 1e-2,
    "score_func": "logistic_accuracy",
    "init": [
        # {
        #     "name": "teleport",
        #     "max_steps": 50,
        #     "allow_sublevel": True,
        #     "line_search": True,
        #     "batch_size": "full_batch",
        #     "lam": 0.1,
        #     "beta": 0.9,
        #     "epochs_to_teleport": [[80, 90]],
        # },
        {"name": "Default"},
    ],
    "opt": [
        {
            "name": "gd_ls",
            "lr": 1e-3,
            "weight_decay": 0,
            "alpha": 0.5,
            "beta": 0.8,
        },
        {
            "name": "newton-cg",
            "lr": [1e-3, 1e-2, 1e-1],
            "cg_max_iter": 10,
            # "line_search": "strong-wolfe",
            "line_search": "none",
            # "disp": True,
            "max_iter": 1,
            "xtol": 1e-20,
            "twice_diffable": True,
        },
    ],
    "batch_size": "full_batch",
    "metric_freq": 5,
    "max_epoch": 100,
    "repeat": list(range(1)),
}


EXPERIMENTS: dict[str, list] = {
    "test": [uci_template],
}
