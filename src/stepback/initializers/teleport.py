"""Sequential linear programming solver for level set teleportation.
"""

import time
from logging import Logger, root, INFO

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from tqdm import tqdm
import numpy as np


# global parameters

ALPHA = 0.5
BETA = 0.5
BETA_INV = 2
GRAD_TOL = 1e-6
CONST_TOL = 1e-6
MAX_BACKTRACKS = 25
GAMMA_SCALE = 0.1
MIN_RHO = 1e-16


def normalized_slp(
    logger: Logger,
    model,
    closure_fn,
    max_steps,
    rho,
    allow_sublevel=False,
    line_search=False,
    alpha=ALPHA,
    max_backtracks=MAX_BACKTRACKS,
    beta=BETA,
):
    """Teleport by solving successive linear approximations.

    This version maximizes the log of the gradient norm, which leads to a
    scale-invariant method and a more stable line-search.

    Params:
        model: the starting model to be teleported.
        obj_fn: callable function which evaluates the objective.
            Must support backward passes.
        max_steps: the maximum number of steps to run the linear SQP method.
        lam: the step-size to use for each step of linear SQP.
        allow_sublevel: whether or not to permit the function value
            to decrease, meaning a sub-level set constraint is used instead of
            a level set constraint.
        line-search: whether or not to use a line-search to set the step-size.
            If True, the value of lam acts as the initial step-size.

    Returns:
        x: approximate solution to teleportation problem.
    """

    if alpha is None:
        alpha = ALPHA

    if max_backtracks is None:
        max_backtracks = MAX_BACKTRACKS

    if beta is None:
        beta = BETA

    beta_inv = 1 / beta

    # metrics
    constr_violations = []
    objectives = []
    kkt_gaps = []
    times = []
    verbose = root.level <= INFO

    parameter_list = list(model.parameters())

    x0 = parameters_to_vector(parameter_list).detach().clone()

    # parameterize network with vector
    x = x0
    obj_fn, _ = closure_fn()

    with torch.no_grad():
        f0, _ = obj_fn(x0, compute_grad=False)
        f0 = f0.item()

    rho0 = rho
    gamma = torch.tensor([0], device=x0.device)

    f_next = None
    grad_next = None

    f_diff_next = None
    g_next = None

    if allow_sublevel:
        penalty_fn = lambda z: torch.maximum(z, torch.tensor([0], device=z.device))
    else:
        penalty_fn = torch.abs

    logger.info("Starting teleportation.")

    for t in tqdm(range(max_steps), disable=not verbose):
        start = time.time()
        obj_fn, hvp_fn = closure_fn()

        if f_next is None:
            func_out, grad = obj_fn(x)
        else:
            # use computations from line-search
            func_out = f_next
            grad = grad_next

        if torch.isnan(func_out) or torch.isinf(func_out):
            logger.warning("Teleportation failed! Returning...")
            return {
                "tele_constr_violation": constr_violations,
                "tele_objective": objectives,
                "tele_kkt_gap": kkt_gaps,
                "tele_time": times,
            }

        q = hvp_fn(x, grad)

        with torch.no_grad():
            if f_diff_next is None:
                f_diff = func_out - f0
                g = grad @ grad
            else:
                g = g_next
                f_diff = f_diff_next

            Hg = q / g
            gHg = torch.inner(Hg, grad)
            gHg_g = gHg / g

            # check termination conditions
            proj = gHg_g * grad - Hg
            kkt_gap = proj @ proj
            if kkt_gap <= GRAD_TOL and penalty_fn(f_diff) <= CONST_TOL:
                logger.warning(
                    "KKT conditions approximately satisfied. Terminating SLP procedure."
                )
                return {
                    "tele_constr_violation": constr_violations,
                    "tele_objective": objectives,
                    "tele_kkt_gap": kkt_gaps,
                    "tele_time": times,
                }

            if verbose:
                tqdm.write(
                    (
                        f"Iteration {t+1}/{max_steps}: "
                        f"Objective: {g.detach().item()}, "
                        f"Constraint Gap: {f_diff.detach().item()}, "
                        f"KKT Residual: {kkt_gap.detach().item()}, "
                        f"Step-size: {rho}"
                    )
                )
            objectives.append(g.detach().item())
            constr_violations.append(f_diff.detach().item())
            kkt_gaps.append(kkt_gap.detach().item())

            # housekeeping for line-search
            x_prev = x.detach().clone()

        for i in range(max_backtracks):
            # evaluate update
            with torch.no_grad():
                x.add_(Hg, alpha=rho)

                v_scale = (rho * gHg + f_diff) / g
                if not allow_sublevel or v_scale > 0:
                    x.sub_(grad, alpha=v_scale.item())

            if not line_search:
                # accept step-size immediately
                break

            # estimate penalty strength for line-search merit function
            gamma = GAMMA_SCALE
            if penalty_fn(f_diff) > 0:
                gamma *= q @ grad * v_scale / (g**2 * penalty_fn(f_diff))

            # proceed with line-search
            f_next, grad_next = obj_fn(x)

            with torch.no_grad():
                # quantities will be re-used for next step
                # if the step-size is accepted.
                g_next = grad_next @ grad_next
                f_diff_next = f_next - f0
                d_t = x - x_prev

                LHS = torch.log(g_next) / 2 - gamma * penalty_fn(f_diff_next)
                RHS = (
                    torch.log(g) / 2
                    - (1 - ALPHA) * gamma * penalty_fn(f_diff)
                    + ALPHA * Hg @ d_t / 2
                )

                if LHS >= RHS:
                    break

                # reset and try with smaller step-size.
                rho = rho * beta
                x[:] = x_prev[:]

        # report if line-search failed
        if i == max_backtracks - 1:
            logger.warning("WARNING: Line-search failed to return feasible step-size.")
            rho = rho * (beta_inv**5)  # try increasing step-size

        if rho <= MIN_RHO:
            logger.warning(
                "WARNING: Cannot find suitable step-size! Returning current iterate..."
            )

            return {
                "tele_constr_violation": constr_violations,
                "tele_objective": objectives,
                "tele_kkt_gap": kkt_gaps,
                "tele_time": times,
            }

        # try to increase step-size if merit bound isn't too tight.
        if line_search and LHS / RHS >= 5.0:
            rho = rho * beta_inv

        dur = time.time() - start
        times.append(dur)

    vector_to_parameters(x.detach(), parameter_list)

    return {
        "tele_constr_violation": constr_violations,
        "tele_objective": objectives,
        "tele_kkt_gap": kkt_gaps,
        "tele_time": times,
    }
