from copy import deepcopy

step_sizes = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

full_mnist_template = {
    "dataset": "mnist",
    "dataset_kwargs": {"as_image": False},
    "model": ["mlp", "smooth_mlp"],
    "model_kwargs": {
        "output_size": 10,
        "hidden_sizes": [[500]],
        "bias": True,
    },
    "loss_func": "cross_entropy",
    "score_func": "cross_entropy_accuracy",
    "reg": 1e-2,
    "init": [
        {
            "name": "teleport",
            "max_steps": 50,
            "allow_sublevel": True,
            "line_search": True,
            "rho": 0.1,
        },
        {"name": "Default"},
    ],
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
    "max_epoch": 10,
    "repeat": list(range(3)),
}
full_mnist_teleport = []

configurations = []

teleport_schedule_stoc = [1, 3, 5, 7, 9]
teleport_schedule_det = [5, 15, 25, 35, 45]
configurations = [
    (128, 40, 10, teleport_schedule_stoc),
    ("full_batch", 1, 50, teleport_schedule_det),
]

for batch_size, eval_freq, epochs, epochs_to_teleport in configurations:
    template_copy = deepcopy(full_mnist_template)
    template_copy["batch_size"] = batch_size
    template_copy["eval_freq"] = eval_freq
    template_copy["max_epoch"] = epochs
    template_copy["init"][0]["epochs_to_teleport"] = [epochs_to_teleport]
    full_mnist_teleport.append(template_copy)

EXPERIMENTS: dict[str, list] = {
    "figures_5_13_14": full_mnist_teleport,
}
