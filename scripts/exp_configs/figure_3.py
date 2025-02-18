from copy import deepcopy

teleport_metrics = {
    "dataset": "mnist",
    "dataset_kwargs": {"as_image": False},
    "model": "smooth_mlp",
    "model_kwargs": {
        "output_size": 10,
        "hidden_sizes": [[50]],
        "bias": True,
    },
    "loss_func": "cross_entropy",
    "score_func": "cross_entropy_accuracy",
    "reg": 1.8,
    "init": [
        {
            "name": "teleport",
            "max_steps": 1000,
            "allow_sublevel": True,
            "line_search": True,
            "lam": 1,
            "alpha": [0.5],
        },
    ],
    "opt": [
        {
            "name": "sgd-m",
            "lr": 1,
            "weight_decay": 0,
            "momentum": 0.9,
            "dampening": 0.9,
            "lr_schedule": "constant",
        },
    ],
    "max_epoch": 1,
    "eval_freq": 1,
    "batch_size": "full_batch",
    "repeat": list(range(3)),
}

EXPERIMENTS: dict[str, list] = {
    "figure_3": [teleport_metrics],
}
