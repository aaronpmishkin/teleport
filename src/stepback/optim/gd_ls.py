"""
Author: Fabian Schaipp

Adapted from https://github.com/fabian-sp/ProxSPS/blob/main/sps/sps.py.

Main changes:
    * use .data in all computations
    * rename 'fstar' to 'lb'
"""

import torch
import warnings

from ..types import Params, LossClosure, OptFloat

MAX_BACKTRACKS = 100
DEFAULT_LR = 1e-8


class GD_LS(torch.optim.Optimizer):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        weight_decay: float = 0,
        alpha: float = 0.5,
        beta: float = 0.8,
    ) -> None:
        """

        Parameters
        ----------
        params :
            PyTorch model parameters.
        lr : float, optional
            Initial learning rate.
        weight_decay : float, optional
            Weigt decay parameter. The default is 0.
            If specified, the term weight_decay/2 * ||w||^2 is added to objective, where w are all model weights.
        alpha: bool, optional
            Tuning parameter for line-search.
        beta: Backtracking parameter.
        """

        params = list(params)
        defaults = dict(lr=lr, weight_decay=weight_decay)

        super(GD_LS, self).__init__(params, defaults)
        self.params = params

        if weight_decay != 0:
            raise ValueError("GD LS does not support implicit weight decay.")
        self.lr = lr
        self.alpha = alpha
        self.closure_nb = None
        self.beta = beta
        self.beta_inv = 1 / beta

        self.current_iter = []
        self.current_grad = []

        self.state["step_size_list"] = list()

        if len(self.param_groups) > 1:
            warnings.warn("More than one parameter group for SPS.")

        return

    def try_update(self, eta):
        for i, p in enumerate(self.params):
            p.data = self.current_iter[i] - eta * self.current_grad[i]

    def set_current_grad_iter(self):
        self.current_iter = []
        self.current_grad = []

        for i, p in enumerate(self.params):
            self.current_iter.append(p.data.clone())
            self.current_grad.append(p.grad.clone())

    def ls_crit(self, f, grad_norm, eta):
        f_prime = self.closure_nb()

        return f_prime <= f - eta * self.alpha * grad_norm

    def step(self, closure: LossClosure = None) -> OptFloat:
        """
        GD with Armijo line-search.
        """

        with torch.enable_grad():
            loss = closure()

        grad_norm = self.compute_grad_norm()

        with torch.no_grad():
            self.set_current_grad_iter()
            self.try_update(self.lr)
            n_backtracks = 0

            # find step-size for which criterion fails:
            while (
                self.ls_crit(loss, grad_norm, self.lr) and n_backtracks < MAX_BACKTRACKS
            ):
                self.lr = self.lr * self.beta_inv
                self.try_update(self.lr)
                n_backtracks += 1

            # Criterion failed, so backtrack once and check.
            self.lr = self.lr * self.beta
            self.try_update(self.lr)
            n_backtracks = 0

            while (
                not self.ls_crit(loss, grad_norm, self.lr)
                and n_backtracks < MAX_BACKTRACKS
            ):
                self.lr = self.lr * self.beta
                self.try_update(self.lr)
                n_backtracks += 1

            if n_backtracks == MAX_BACKTRACKS:
                self.lr = DEFAULT_LR
                self.try_update(self.lr)

        # update state with metrics
        self.state["step_size_list"].append(self.lr)  # works only if one param_group!

        return loss

    @torch.no_grad()
    def compute_grad_norm(self):
        grad_norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    raise KeyError("None gradient")

                g = p.grad.data
                grad_norm += torch.sum(torch.mul(g, g))

        return grad_norm

    def set_vectorized_closure(self, vectorized_closure):
        self.vectorized_closure = vectorized_closure
