import warnings

import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR
from torchmin import Minimizer

from .momo import Momo

from .momo_adam import MomoAdam
from .sps import SPS
from .normalized_sgd import NormalizedSGD
from .gd_ls import GD_LS


def get_optimizer(opt_config: dict) -> (torch.optim.Optimizer, dict):
    """
    Main function mapping opt configs to an instance of torch.optim.Optimizer and a dict of hyperparameter arguments (lr, weight_decay,..).

    For all hyperparameters which are not specified, we use PyTorch default.
    """

    name = opt_config["name"]

    if opt_config.get("lr") is None:
        warnings.warn(
            "You have not specified a learning rate. A default value of 1e-3 will be used."
        )

    if name == "sgd":
        opt_obj = torch.optim.SGD

        hyperp = {
            "lr": opt_config.get("lr", 1e-3),
            "weight_decay": opt_config.get("weight_decay", 0),
        }

    elif name == "normalized-sgd":
        opt_obj = NormalizedSGD

        hyperp = {
            "lr": opt_config.get("lr", 1e-3),
            "weight_decay": opt_config.get("weight_decay", 0),
        }

    elif name == "gd_ls":
        opt_obj = GD_LS

        hyperp = {
            "lr": opt_config.get("lr", 1e-3),
            "weight_decay": opt_config.get("weight_decay", 0),
            "alpha": opt_config.get("alpha", 0.5),
            "beta": opt_config.get("beta", 0.8),
        }

    elif name == "sgd-m":
        opt_obj = torch.optim.SGD
        # sgd-m with exp. weighted average should have dampening = momentum
        if opt_config.get("dampening") == "momentum":
            dampening = opt_config.get("momentum", 0.9)
        else:
            dampening = opt_config.get("dampening", 0)

        hyperp = {
            "lr": opt_config.get("lr", 1e-3),
            "weight_decay": opt_config.get("weight_decay", 0),
            "momentum": opt_config.get("momentum", 0.9),
            "nesterov": False,
            "dampening": dampening,
        }

    elif name == "sgd-nesterov":
        opt_obj = torch.optim.SGD
        hyperp = {
            "lr": opt_config.get("lr", 1e-3),
            "weight_decay": opt_config.get("weight_decay", 0),
            "momentum": opt_config.get("momentum", 0.9),
            "nesterov": True,
            "dampening": opt_config.get("dampening", 0),
        }

    elif name == "adam":
        opt_obj = torch.optim.Adam
        hyperp = {
            "lr": opt_config.get("lr", 1e-3),
            "weight_decay": opt_config.get("weight_decay", 0),
            "betas": opt_config.get("betas", (0.9, 0.999)),
            "eps": opt_config.get("eps", 1e-8),
        }

    elif name == "adamw":
        opt_obj = torch.optim.AdamW
        hyperp = {
            "lr": opt_config.get("lr", 1e-3),
            "weight_decay": opt_config.get("weight_decay", 0),
            "betas": opt_config.get("betas", (0.9, 0.999)),
            "eps": opt_config.get("eps", 1e-8),
        }

    elif name == "momo":
        opt_obj = Momo
        hyperp = {
            "lr": opt_config.get("lr", 1e-3),
            "weight_decay": opt_config.get("weight_decay", 0),
            "beta": opt_config.get("beta", 0.9),
            "lb": opt_config.get("lb", 0.0),
            "bias_correction": opt_config.get("bias_correction", False),
            "use_fstar": False,
        }

    elif name == "momo-adam":
        opt_obj = MomoAdam
        hyperp = {
            "lr": opt_config.get("lr", 1e-3),
            "weight_decay": opt_config.get("weight_decay", 0),
            "betas": opt_config.get("betas", (0.9, 0.999)),
            "eps": opt_config.get("eps", 1e-8),
            "lb": opt_config.get("lb", 0.0),
            "divide": opt_config.get("divide", True),
            "use_fstar": False,
        }

    elif name == "momo-star":
        opt_obj = Momo
        hyperp = {
            "lr": opt_config.get("lr", 1e-3),
            "weight_decay": opt_config.get("weight_decay", 0),
            "beta": opt_config.get("beta", 0.9),
            "lb": opt_config.get("lb", 0.0),
            "bias_correction": opt_config.get("bias_correction", False),
            "use_fstar": True,
        }

    elif name == "momo-adam-star":
        opt_obj = MomoAdam
        hyperp = {
            "lr": opt_config.get("lr", 1e-3),
            "weight_decay": opt_config.get("weight_decay", 0),
            "betas": opt_config.get("betas", (0.9, 0.999)),
            "eps": opt_config.get("eps", 1e-8),
            "lb": opt_config.get("lb", 0.0),
            "divide": opt_config.get("divide", True),
            "use_fstar": True,
        }

    elif name == "sps":
        opt_obj = SPS
        hyperp = {
            "lr": opt_config.get("lr", 1e-3),
            "weight_decay": opt_config.get("weight_decay", 0),
            "lb": opt_config.get("lb", 0.0),
            "prox": opt_config.get("prox", False),
        }
    elif name == "newton-cg":
        opt_obj = Minimizer
        hyperp = {
            "method": "newton-cg",
            "options": {
                "lr": opt_config.get("lr", 1e-3),
                "cg_max_iter": opt_config.get("cg_max_iter", 10),
                "line_search": opt_config.get("line_search", "strong-wolfe"),
                "twice_diffable": opt_config.get("twice_diffable", False),
                "disp": opt_config.get("disp", False),
                "max_iter": opt_config.get("max_iter", 1),
                "xtol": opt_config.get("xtol", 1e-10),
            },
        }

    else:
        raise KeyError(f"Unknown optimizer name {name}.")

    return opt_obj, hyperp


def get_scheduler(
    config: dict, opt: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Main function mapping to a learning rate scheduler.
    """
    # if not specified, use constant step sizes
    name = config.get("lr_schedule", None)

    if name == "constant":
        lr_fun = lambda epoch: 1  # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)

    elif name == "linear":
        lr_fun = lambda epoch: 1 / (
            epoch + 1
        )  # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)

    elif name == "sqrt":
        lr_fun = lambda epoch: (epoch + 1) ** (
            -1 / 2
        )  # this value is multiplied with initial lr
        scheduler = LambdaLR(opt, lr_lambda=lr_fun)

    elif name == "exponential":
        # TODO: allow arguments
        scheduler = StepLR(opt, step_size=50, gamma=0.5)

    elif name == "multi_step":
        milestones = config.get("lr_milestones", [])
        gamma = config.get("lr_gamma", 1)
        scheduler = MultiStepLR(opt, milestones=milestones, gamma=gamma)

    elif name is None:
        scheduler = None

    else:
        raise ValueError(f"Unknown learning rate schedule name {name}.")

    return scheduler
