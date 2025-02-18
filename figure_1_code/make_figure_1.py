from functools import partial
import time

import torch
from tqdm import tqdm
import numpy as np
import pybenchfunction as bench

from src.stoch_functions import (
    LevyN13_i,
    PermDBeta_i,
    Rastrigin_i,
    RosenBrock_i,
)
from src.torch_functions import (
    Rosenbrock,
    Rastrigin,
    IllQuad,
    Booth,
    BukinN6,
    GoldsteinPrice,
    Himmelblau,
)
from src.teleport import (
    normalized_slp,
    identity,
)
from src.algorithms import run_GD_teleport, run_newton
from src.plotting import plot_function_values, plot_level_set_results


def run_methods(
    x0,
    func,
    bench_func,
    stepsize=0.001,
    epochs=500,
    teleport_num=100,
    teleport_lr=10**-1,
    teleport_lr_norm=10**-1,
    teleport_steps=3,
    logscale=False,
):

    # line-search version
    sqp_teleport_ls = partial(
        normalized_slp,
        max_steps=teleport_steps,
        rho=10,
        verbose=True,
        line_search=True,
        allow_sublevel=True,
    )
    t0 = time.perf_counter()
    gd_ls_tp_x_list, gd_ls_tp_fval, gd_ls_tp_path = run_GD_teleport(
        func,
        sqp_teleport_ls,
        epochs=epochs,
        x0=x0,
        d=d,
        lr=stepsize,
        teleport_num=teleport_num,
    )
    gd_ls_tp_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    gd_x_list, gd_fval, _ = run_GD_teleport(
        func,
        identity,
        epochs=epochs,
        x0=x0,
        d=d,
        lr=stepsize,
    )
    gd_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    newt_x_list, newt_fval = run_newton(func, epochs=20, x0=x0, d=d, lr=0.99)
    newt_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    results = {
        "GD (Teleport)": (
            gd_ls_tp_time,
            gd_ls_tp_fval,
            gd_ls_tp_x_list,
            gd_ls_tp_path,
        ),
        "GD": (gd_time, gd_fval, gd_x_list, []),
        "Newton": (newt_time, newt_fval, newt_x_list, []),
    }
    plot_level_set_results(bench_func, results, show=False, logscale=logscale)


teleport_steps = 1000

d = 2
lr = 1
x0 = torch.tensor([7.5, -4.0], requires_grad=True).double()  # teleport_steps=5
run_methods(
    x0,
    Booth,
    bench.function.Booth(d),
    stepsize=1,
    epochs=100,
    teleport_num=1000,
    teleport_lr=1e-2,
    teleport_lr_norm=1e1,
    teleport_steps=teleport_steps,
)


x0 = torch.tensor([0.0, 0.0], requires_grad=True).double()  # teleport_steps=5
run_methods(
    x0,
    GoldsteinPrice,
    bench.function.GoldsteinPrice(d),
    stepsize=1,
    epochs=100,
    teleport_num=1000,
    teleport_lr=1e-9,
    teleport_lr_norm=1e-2,
    teleport_steps=teleport_steps,
    logscale=True,
)
