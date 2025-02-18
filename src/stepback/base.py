import copy
import time
import datetime
import warnings
from collections import defaultdict
from logging import Logger, root, INFO

import numpy as np
import tqdm  # type: ignore
import torch
from torchmin import Minimizer

from torch.utils.data import DataLoader, TensorDataset

from .datasets.main import get_dataset, infer_shapes, DataClass, BatchLoader
from .models.main import get_model
from .optim.main import get_optimizer, get_scheduler
from .initializers.main import get_initializer
from .metrics import Loss
from .utils import update_momentum, parameter_norm

from .utils import (
    l2_norm,
    grad_norm,
    ridge_opt_value,
    logreg_opt_value,
    vector_to_parameters,
)


class Base:
    def __init__(
        self,
        logger: Logger,
        name: str,
        config: dict,
        device: str = "cpu",
        data_dir: str = "data/",
    ):
        self.name = name
        self.config = copy.deepcopy(config)
        self.data_dir = data_dir
        self.verbose = root.level <= INFO
        self.logger = logger

        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        self.logger.info(f"Using device: {self.device}")

        repeat = config.get("repeat", 0)
        self.seed = 1234567 + repeat
        self.run_seed = 456789 + repeat
        torch.backends.cudnn.benchmark = False
        # see https://pytorch.org/docs/stable/notes/randomness.html#reproducibility

        self.check_config()

    def check_config(self):
        # create defaults for missing config keys
        if "batch_size" not in self.config.keys():
            self.config["batch_size"] = 1

        if "dataset_kwargs" not in self.config.keys():
            self.config["dataset_kwargs"] = dict()

        if "model_kwargs" not in self.config.keys():
            self.config["model_kwargs"] = dict()

        if "init" not in self.config.keys():
            self.config["init"] = dict()

        # check necessary config keys
        for k in ["loss_func", "score_func", "opt"]:
            assert (
                k in self.config.keys()
            ), f"You need to specify {k} in the config file."

        return

    def _setup_data(self):
        """Loads training and validation set. Creates DataLoader for training."""
        self.train_set = get_dataset(
            config=self.config,
            split="train",
            seed=self.seed,
            path=self.data_dir,
        )
        self.val_set = get_dataset(
            config=self.config,
            split="val",
            seed=self.seed,
            path=self.data_dir,
        )

        self.config["_input_dim"], self.config["_output_dim"] = infer_shapes(
            self.train_set
        )

        if self.config["batch_size"] == "full_batch":
            self.batch_size = len(self.train_set)

            dl = torch.utils.data.DataLoader(
                self.train_set,
                drop_last=False,
                batch_size=len(self.train_set),
            )

            data, targets = next(iter(dl))
            data = data.to(self.device)
            targets = targets.to(self.device)
            train_set = DataClass(TensorDataset(data, targets), split="train")

            self.train_loader = BatchLoader(train_set, (data, targets))

            val_dl = torch.utils.data.DataLoader(
                self.val_set,
                drop_last=False,
                batch_size=len(self.val_set),
            )

            data, targets = next(iter(val_dl))
            data = data.to(self.device)
            targets = targets.to(self.device)
            val_set = DataClass(TensorDataset(data, targets), split="val")

            self.val_loader = BatchLoader(val_set, (data, targets))

        else:
            self.batch_size = self.config["batch_size"]
            train_set = self.train_set
            val_set = self.val_set

            # construct train loader
            _gen = torch.Generator()
            _gen.manual_seed(self.run_seed)
            self.train_loader = DataLoader(
                train_set,
                drop_last=True,
                shuffle=True,
                generator=_gen,
                batch_size=self.batch_size,
            )

            self.val_loader = torch.utils.data.DataLoader(
                val_set, drop_last=False, batch_size=self.batch_size
            )

    def _setup_model(self):
        """Initializes the model."""
        torch.manual_seed(self.seed)  # Reseed to have same initialization
        torch.cuda.manual_seed_all(self.seed)

        self.model = get_model(self.config)
        self.model.to(self.device)

    def setup(self):
        # ============ Data =================
        self._setup_data()

        # ============ Model ================
        self._setup_model()
        self.logger.info(self.model)

        # ============ Loss function ========

        # optional weight decay regularization
        reg = self.config.get("reg", 0.0)
        clear_grad = self.config["opt"].get("clear_grad", False)
        self.training_loss = Loss(
            name=self.config["loss_func"],
            backwards=True,
            clear_grad=clear_grad,
            reg=reg,
        )

        self.training_loss_nb = Loss(
            name=self.config["loss_func"],
            backwards=False,
            clear_grad=clear_grad,
            reg=reg,
        )

        # frequency at which metrics are recorded
        self.eval_freq = self.config.get("eval_freq", "epoch")

        if self.eval_freq == "epoch":
            self.eval_freq = np.ceil(len(self.train_set) / self.batch_size)

        # ============ Initializer ==========

        self.init_fn, _ = get_initializer(self.config["init"])
        init_config = self.config["init"]
        self.beta = init_config.get("beta", 0.9)
        self.teleport_on_demand = init_config.get("on_demand", False)
        self.teleport_tol = init_config.get("tol", 1e-2)
        self.teleport_wait = init_config.get("wait", 100)
        self.epochs_to_teleport = init_config.get("epochs_to_teleport", [])
        self.last_teleport = 0
        self.cum_teleports = 0

        # ============ Optimizer ============
        opt_obj, hyperp = get_optimizer(self.config["opt"])

        self._init_opt(opt_obj, hyperp)

        self.sched = get_scheduler(self.config["opt"], self.opt)

    def _init_opt(self, opt_obj, hyperp):
        """Initializes the opt object. If your optimizer needs custom commands, add them here."""

        self.opt = opt_obj(params=self.model.parameters(), **hyperp)

        self.logger.info(self.opt)

    def run(self):
        start_time = str(datetime.datetime.now())
        self._epochs_trained = 0

        metrics = defaultdict(list)

        # initialize model
        if not self.teleport_on_demand and len(self.epochs_to_teleport) == 0:
            init_metrics = self.initialize_model()
            metrics.update(init_metrics)

        s_time = time.time()
        # momentum estimator
        m = None
        metrics["epoch"].append(0)
        iter_total = 0
        metrics["iter"] = []
        for epoch in range(self.config["max_epoch"]):
            # Train one epoch
            metrics, iter_total, s_time = self.train_epoch(
                metrics, iter_total, s_time, m
            )
            metrics["epoch"].append(epoch + 1)

            self._epochs_trained += 1

            if epoch in self.epochs_to_teleport:
                self.logger.info("Triggering teleport.")
                self.initialize_model()
                self.cum_teleports += 1

        end_time = str(datetime.datetime.now())

        return (
            {"start_time": start_time, "end_time": end_time, "success": True},
            self.model,
            metrics,
        )

    def record_metrics(self, metrics, time_diff):
        # Record metrics
        metrics["train_epoch_time"].append(time_diff)
        metrics["model_norm"].append(l2_norm(self.model))
        metrics["grad_norm"].append(grad_norm(self.model))

        # Validation
        with torch.no_grad():
            metric_funcs = {
                "loss": Loss(self.config["loss_func"], backwards=False, clear_grad=False),
                "score": Loss(self.config["score_func"], backwards=False, clear_grad=False),
            }

            train_dict = self.evaluate(
                self.train_loader,
                metric_dict=metric_funcs,
            )

            for k, v in train_dict.items():
                metrics[k].append(v)

            val_dict = self.evaluate(
                self.val_loader,
                metric_dict=metric_funcs,
            )

            for k, v in val_dict.items():
                metrics[k].append(v)

        return metrics

    def initialize_model(self):
        batch_size = self.config["init"].get("batch_size", "full_batch")

        # wrap data inside objective
        def compute_batch_obj(x, batch, compute_grad=True):
            data, targets = batch
            data = data.to(device=self.device)
            targets = targets.to(device=self.device)

            vector_to_parameters(x, init_model.parameters())
            out = init_model(data)
            batch_obj = self.training_loss_nb.compute(out, targets, init_model)

            if compute_grad:
                batch_grad = torch.autograd.grad(batch_obj, x)[0]
            else:
                batch_grad = 0

            return batch_obj, batch_grad

        def get_closures(batch):
            def obj_fn(x, compute_grad=True):
                y = x.detach()
                y.requires_grad = True
                return compute_batch_obj(y, batch, compute_grad)

            def hvp_function(x, v):
                def obj_wrapper(x):
                    batch_obj, _ = compute_batch_obj(x, batch, False)
                    return batch_obj

                Hv = torch.autograd.functional.hvp(
                    obj_wrapper,
                    x,
                    v,
                )[1]

                return Hv

            return obj_fn, hvp_function

        if batch_size == "full_batch":
            dl = torch.utils.data.DataLoader(
                self.train_set,
                drop_last=False,
                batch_size=len(self.train_set),
            )

            data, targets = next(iter(dl))
            full_batch = data.to(self.device), targets.to(self.device)

            def closure_fn():
                return get_closures(full_batch)

        else:
            dl = torch.utils.data.DataLoader(
                self.train_set,
                drop_last=False,
                batch_size=batch_size,
            )

            def closure_fn():
                batch = next(iter(dl))
                return get_closures(batch)

        init_model = get_model(self.config)
        init_model.to(self.device)

        for p, p_init in zip(self.model.parameters(), init_model.parameters()):
            # detach init model parameters from graph
            p_init.requires_grad = False
            # copy parameter state from main model.
            p_init.data = p.data.detach()

        metrics = self.init_fn(self.logger, init_model, closure_fn)

        # reattach parameters to graph
        for p, p_init in zip(self.model.parameters(), init_model.parameters()):
            p.data = p_init.data.detach()

        return metrics

    def train_epoch(self, metrics, iter_total, s_time, m):
        """
        Train one epoch.
        """

        self.model.train()
        pbar = tqdm.tqdm(self.train_loader, disable=not self.verbose)

        for data, targets in pbar:
            if iter_total % self.eval_freq == 0 or self.last_teleport == iter_total:
                e_time = time.time()
                metrics = self.record_metrics(metrics, e_time - s_time)
                metrics["iter"].append(iter_total)

                if m is not None:
                    metrics["momentum_norm"].append(parameter_norm(m))
                    metrics["cum_teleports"].append(self.cum_teleports)
                    metrics["teleport_iters"].append(self.last_teleport)

                s_time = time.time()

            iter_total += 1

            self.opt.zero_grad()

            # get batch and compute model output
            data, targets = data.to(device=self.device), targets.to(device=self.device)
            out = self.model(data).to(device=self.device)

            if len(out.shape) <= 1:
                warnings.warn(
                    f"Shape of model output is {out.shape}, recommended to have shape [batch_size, ..]."
                )

            closure = lambda: self.training_loss.compute(
                None, targets, self.model, recompute=True, data=data
            )
            closure_nb = lambda: self.training_loss_nb.compute(
                None, targets, self.model, recompute=True, data=data
            )
            self.opt.closure_nb = closure_nb

            if isinstance(self.opt, Minimizer):
                loss_val = self.opt.step(closure=closure_nb)
            else:
                loss_val = self.opt.step(closure=closure)

            pbar.set_description(f"Training - {loss_val:.3f}")

            # check for teleportation
            if self.teleport_on_demand:
                m = update_momentum(self.model.parameters(), m, self.beta)
                momentum_norm = parameter_norm(m)

                if (
                    momentum_norm < self.teleport_tol
                    and iter_total - self.last_teleport > self.teleport_wait
                ):
                    self.logger.info("Triggering teleport.")
                    self.initialize_model()
                    self.last_teleport = iter_total
                    self.cum_teleports += 1

        # update learning rate
        if self.sched is not None:
            self.sched.step()
        return metrics, iter_total, s_time

    def evaluate(self, loader, metric_dict):
        """
        Evaluate model for a given dataset (train or val), and for several metrics.

        metric_dict:
            Should have the form {'metric_name1': metric1, 'metric_name2': metric2, ...}
        """

        pbar = tqdm.tqdm(loader, disable=not self.verbose)

        self.model.eval()
        score_dict = dict(zip(metric_dict.keys(), np.zeros(len(metric_dict))))

        for data, targets in pbar:
            # get batch and compute model output
            data, targets = data.to(device=self.device), targets.to(device=self.device)
            out = self.model(data)

            for _met, _met_fun in metric_dict.items():
                # metric takes average over batch ==> multiply with batch size
                score_dict[_met] += (
                    _met_fun.compute(out, targets).item() * data.shape[0]
                )

            pbar.set_description(f"Validating {loader.dataset.split}")

        for _met in metric_dict.keys():
            # Get from sum to average
            score_dict[_met] = float(score_dict[_met] / len(loader.dataset))

            # add split in front of names
            score_dict[loader.dataset.split + "_" + _met] = score_dict.pop(_met)

        self.model.train()

        return score_dict

    def save_checkpoint(self, path):
        """See https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html"""
        torch.save(
            {
                "epoch": self._epochs_trained,
                "model_state_dict": self.model.state_dict(),
                "opt_state_dict": self.opt.state_dict(),
            },
            path + self.name + ".mt",
        )

    def _compute_opt_value(self):
        """
        For linear model, the problem is convex and we can compute the optimal value
        """
        if self.config["model"] == "linear":
            # fit_intercept = (self.model[0].bias is not None)

            if self.config["loss_func"] == "squared":
                opt_val = ridge_opt_value(
                    X=self.train_set.dataset.tensors[0].detach().numpy(),
                    y=self.train_set.dataset.tensors[1].detach().numpy(),
                    lmbda=self.config["opt"].get("weight_decay", 0),
                    fit_intercept=False,
                )
            elif self.config["loss_func"] == "logistic":
                opt_val = logreg_opt_value(
                    X=self.train_set.dataset.tensors[0].detach().numpy(),
                    y=self.train_set.dataset.tensors[1]
                    .detach()
                    .numpy()
                    .astype(int)
                    .reshape(-1),
                    lmbda=self.config["opt"].get("weight_decay", 0),
                    fit_intercept=False,
                )
            else:
                opt_val = None
        else:
            opt_val = None

        return opt_val
