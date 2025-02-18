from typing import Callable

import torch

from .teleport import normalized_slp


def get_initializer(init_config: dict) -> (Callable, dict):
    """
    Main function mapping initialization configs to a callable initializer
    and dict of hyperparameters for the initializer.
    """

    name = init_config.get("name", None)

    if name == "teleport":
        max_steps = init_config.get("max_steps", 100)
        rho = init_config.get("rho", 1)
        allow_sublevel = init_config.get("allow_sublevel", True)
        line_search = init_config.get("line_search", True)
        alpha = init_config.get("alpha", None)
        max_backtracks = init_config.get("max_backtracks", None)
        beta = init_config.get("beta", None)

        def init_fn(logger, model, closure_fn):
            return normalized_slp(
                logger,
                model,
                closure_fn,
                max_steps,
                rho,
                allow_sublevel,
                line_search,
                alpha,
                max_backtracks,
                beta
            )

        hyperp = {
            "max_steps": max_steps,
            "rho": rho,
            "allow_sublevel": allow_sublevel,
            "line_search": line_search,
        }

    elif name == "uniform":
        a, b = init_config.get("a", 0.0), init_config.get("b", 1.0)

        def init_fn(logger, model, closure_fn):
            for p in model.parameters():
                torch.nn.init.uniform_(p, a=a, b=b)

            return {}

        hyperp = {
            "a": a,
            "b": b,
        }

    elif name == "normal":
        mean, std = init_config.get("mean", 0.0), init_config.get("std", 1.0)

        def init_fn(logger, model, closure_fn):
            for p in model.parameters():
                torch.nn.init.normal_(p, mean=mean, std=std)

            return {}

        hyperp = {
            "mean": mean,
            "std": std,
        }

    elif name == "xavier_uniform":
        gain = init_config.get("gain", 1.0)

        def init_fn(logger, model, closure_fn):
            for p in model.parameters():
                torch.nn.init.xavier_uniform_(p, gain=gain)

            return {}

        hyperp = {
            "gain": gain,
        }

    elif name == "xavier_normal":
        gain = init_config.get("gain", 1.0)

        def init_fn(logger, model, closure_fn):
            for p in model.parameters():
                torch.nn.init.xavier_normal_(p, gain=gain)

            return {}

        hyperp = {
            "gain": gain,
        }

    elif name == "kaiming_uniform":
        a = init_config.get("a", 0.0)
        mode = init_config.get("mode", "fan_in")
        non_linearity = init_config.get("non_linearity", "leaky_relu")

        def init_fn(logger, model, closure_fn):
            for p in model.parameters():
                torch.nn.init.kaiming_uniform_(
                    p,
                    a=a,
                    mode=mode,
                    non_linearity=non_linearity,
                )

            return {}

        hyperp = {
            "a": a,
            "mode": mode,
            "non_linearity": non_linearity,
        }

    elif name == "kaiming_normal":
        a = init_config.get("a", 0.0)
        mode = init_config.get("mode", "fan_in")
        non_linearity = init_config.get("non_linearity", "leaky_relu")

        def init_fn(logger, model, closure_fn):
            for p in model.parameters():
                torch.nn.init.kaiming_normal_(
                    p,
                    a=a,
                    mode=mode,
                    non_linearity=non_linearity,
                )

            return {}

        hyperp = {
            "a": a,
            "mode": mode,
            "non_linearity": non_linearity,
        }
    else:

        def init_fn(logger, model, closure_fn):
            return {}

        hyperp = {}

    return init_fn, hyperp
