import copy
from math import ceil
import numpy as np
from scipy.spatial.distance import cdist
import logging
from typing import Literal
import os
import sys
import time

import torch
from torch.nn import Softmax
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from lambdaLearn.Base.BaseOptimizer import BaseOptimizer
from lambdaLearn.Base.BaseScheduler import BaseScheduler
from lambdaLearn.Base.SemiEstimator import SemiEstimator
from lambdaLearn.Dataloader.TrainDataloader import TrainDataLoader
from lambdaLearn.Dataset.TrainDataset import TrainDataset
from lambdaLearn.utils import EMA, to_device, tensor2numpy
from lambdaLearn.Dataset.CILDataset import CILDataset


class DeepModelMixin(SemiEstimator):
    def __init__(
            self,
            train_dataset=None,
            labeled_dataset=None,
            unlabeled_dataset=None,
            valid_dataset=None,
            test_dataset=None,
            train_dataloader=None,
            labeled_dataloader=None,
            unlabeled_dataloader=None,
            valid_dataloader=None,
            test_dataloader=None,
            augmentation=None,
            network=None,
            epoch=1,
            num_it_epoch=None,
            num_it_total=None,
            eval_epoch=None,
            eval_it=None,
            mu=None,
            optimizer=None,
            weight_decay=5e-4,
            ema_decay=None,
            scheduler=None,
            device=None,
            evaluation=None,
            train_sampler=None,
            labeled_sampler=None,
            unlabeled_sampler=None,
            train_batch_sampler=None,
            labeled_batch_sampler=None,
            unlabeled_batch_sampler=None,
            valid_sampler=None,
            valid_batch_sampler=None,
            test_sampler=None,
            test_batch_sampler=None,
            parallel=None,
            file=None,
            verbose=True,
    ):
        """
        :param train_dataset: Data manager for training data.
        :param labeled_dataset: Data manager for labeled data.
        :param unlabeled_dataset: Data manager for unlabeled data.
        :param valid_dataset: Data manager for valid data.
        :param test_dataset: Data manager for test data.
        :param augmentation: Augmentation method, if there are multiple augmentation methods, you can use a dictionary or a list to pass parameters.
        :param network: The backbone neural network.
        :param epoch: Number of training epochs.
        :param num_it_epoch: The number of iterations in each round, that is, the number of batches of data.
        :param num_it_total: The total number of batches.
        :param eval_epoch: Model evaluation is performed every eval_epoch epochs.
        :param eval_it: Model evaluation is performed every eval_it iterations.
        :param mu: The ratio of the number of unlabeled data to the number of labeled data.
        :param optimizer: The optimizer used in training.
        :param weight_decay: The optimizer's learning rate decay parameter.
        :param ema_decay: The update scale for the exponential moving average of the model parameters.
        :param scheduler: Learning rate scheduler.
        :param device: Training equipment.
        :param evaluation: Model evaluation metrics. If there are multiple metrics, a dictionary or a list can be used.
        :param train_sampler: Sampler of training data.
        :param labeled_sampler=None: Sampler of labeled data.
        :param unlabeled_sampler=None: Sampler of unlabeled data.
        :param train_batch_sampler=None: Batch sampler of training data
        :param labeled_batch_sampler: Batch sampler of labeled data
        :param unlabeled_batch_sampler: Batch sampler of unlabeled data
        :param valid_sampler: sampler of valid data.
        :param valid_batch_sampler: Batch sampler of valid data.
        :param test_sampler: Sampler of test data.
        :param test_batch_sampler: Batch sampler of test data.
        :param parallel: Distributed training method.
        :param file: Output file.
        """
        self.train_dataset = train_dataset
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.valid_dataset = valid_dataset if valid_dataset is not None else test_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = train_dataloader
        self.labeled_dataloader = labeled_dataloader
        self.unlabeled_dataloader = unlabeled_dataloader
        self.valid_dataloader = valid_dataloader if valid_dataloader is not None else test_dataloader
        self.test_dataloader = test_dataloader
        self.augmentation = augmentation
        self.network = network
        self.epoch = epoch
        self.mu = mu
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.eval_epoch = eval_epoch
        self.eval_it = eval_it
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.y_est = None
        self.y_true = None
        self.y_pred = None
        self.y_score = None
        self.num_it_epoch = num_it_epoch
        self.num_it_total = num_it_total
        self.evaluation = evaluation

        self.train_sampler = train_sampler
        self.train_batch_sampler = train_batch_sampler

        self.labeled_sampler = labeled_sampler
        self.unlabeled_sampler = unlabeled_sampler
        self.labeled_batch_sampler = labeled_batch_sampler
        self.unlabeled_batch_sampler = unlabeled_batch_sampler

        self.valid_sampler = valid_sampler if valid_sampler is not None else test_sampler
        self.valid_batch_sampler = valid_batch_sampler if valid_batch_sampler is not None else test_batch_sampler

        self.test_sampler = test_sampler
        self.test_batch_sampler = test_batch_sampler

        self.valid_performance = None

        self.parallel = parallel
        self.verbose = verbose
        self.it_epoch = 0
        self.it_total = 0
        self.loss = None
        self.weak_augmentation = None
        self.strong_augmentation = None
        self.normalization = None
        self.performance = None
        self.valid_performance = None
        self.ema = None
        if isinstance(file, str):
            file = open(file, "w")
        self.file = file
        self._estimator_type = None

    def init_model(self):
        self._network = copy.deepcopy(self.network)
        self._parallel = copy.deepcopy(self.parallel)
        if self.device is None:
            self.device = "cpu"
        if self.device != "cpu":
            torch.cuda.set_device(self.device)
        self._network = self._network.to(self.device)
        if self._parallel is not None:
            self._network = self._parallel.init_parallel(self._network)

    def init_ema(self):
        if self.ema_decay is not None:
            self.ema = EMA(model=self._network, decay=self.ema_decay)
            self.ema.register()
        else:
            self.ema = None

    def init_optimizer(self):
        self._optimizer = copy.deepcopy(self.optimizer)
        if isinstance(self._optimizer, BaseOptimizer):
            no_decay = ["bias", "bn"]
            grouped_parameters = [
                {
                    "params": [p for n, p in self._network.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in self._network.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self._optimizer = self._optimizer.init_optimizer(params=grouped_parameters)

    def init_scheduler(self):
        self._scheduler = copy.deepcopy(self.scheduler)
        if isinstance(self._scheduler, BaseScheduler):
            self._scheduler = self._scheduler.init_scheduler(optimizer=self._optimizer)

    def init_epoch(self):
        if self.num_it_epoch is not None and self.epoch is not None:
            self.num_it_total = self.epoch * self.num_it_epoch
        elif self.num_it_total is not None and self.epoch is not None:
            self.num_it_epoch = ceil(self.num_it_total / self.epoch)
        elif self.num_it_total is not None and self.num_it_epoch is not None:
            self.epoch = ceil(self.num_it_total / self.num_it_epoch)

    def init_augmentation(self):
        self._augmentation = copy.deepcopy(self.augmentation)
        if self._augmentation is not None:
            if isinstance(self._augmentation, dict):
                self.weak_augmentation = (
                    self._augmentation["augmentation"]
                    if "augmentation" in self._augmentation.keys()
                    else self._augmentation["weak_augmentation"]
                )
                if "strong_augmentation" in self._augmentation.keys():
                    self.strong_augmentation = self._augmentation["strong_augmentation"]
            elif isinstance(self._augmentation, (list, tuple)):
                self.weak_augmentation = self._augmentation[0]
                if len(self._augmentation) > 1:
                    self.strong_augmentation = self._augmentation[1]
            else:
                self.weak_augmentation = copy.copy(self._augmentation)
            if self.strong_augmentation is None:
                self.strong_augmentation = copy.copy(self.weak_augmentation)

    def init_transform(self):
        if self.weak_augmentation is not None:
            self._train_dataset.add_transform(self.weak_augmentation, dim=1, x=0, y=0)
            self._train_dataset.add_unlabeled_transform(self.weak_augmentation, dim=1, x=0, y=0)

    def init_train_dataset(self, X=None, y=None, unlabeled_X=None, *args, **kwargs):
        self.train_dataset = (
            self.train_dataset
            if self.train_dataset is not None
            else TrainDataset(
                labeled_dataset=self.labeled_dataset,
                unlabeled_dataset=self.unlabeled_dataset,
            )
        )
        self._train_dataset = copy.deepcopy(self.train_dataset)
        if isinstance(X, TrainDataset):
            self._train_dataset = X
        elif isinstance(X, Dataset) and y is None:
            self._train_dataset.init_dataset(labeled_dataset=X, unlabeled_dataset=unlabeled_X)
        else:
            self._train_dataset.init_dataset(labeled_X=X, labeled_y=y, unlabeled_X=unlabeled_X)

    def init_train_dataloader(self):
        self._train_dataloader = copy.deepcopy(self.train_dataloader)
        self._labeled_dataloader = copy.deepcopy(self.labeled_dataloader)
        self._unlabeled_dataloader = copy.deepcopy(self.unlabeled_dataloader)
        self._train_sampler = copy.deepcopy(self.train_sampler)
        self._labeled_sampler = copy.deepcopy(self.labeled_sampler)
        self._unlabeled_sampler = copy.deepcopy(self.unlabeled_sampler)
        self._train_batch_sampler = copy.deepcopy(self.train_batch_sampler)
        self._labeled_batch_sampler = copy.deepcopy(self.labeled_batch_sampler)
        self._unlabeled_batch_sampler = copy.deepcopy(self.unlabeled_batch_sampler)
        if self._train_dataloader is not None:
            (
                self._labeled_dataloader,
                self._unlabeled_dataloader,
            ) = self._train_dataloader.init_dataloader(
                dataset=self._train_dataset,
                sampler=self._train_sampler,
                batch_sampler=self._train_batch_sampler,
                mu=self.mu,
            )
        else:
            self._train_dataloader = TrainDataLoader(
                labeled_dataloader=self._labeled_dataloader,
                unlabeled_dataloader=self._unlabeled_dataloader,
            )
            self._train_sampler = {
                "labeled": self._labeled_sampler,
                "unlabeled": self._unlabeled_sampler,
            }
            self._train_batch_sampler = {
                "labeled": self._labeled_batch_sampler,
                "unlabeled": self._unlabeled_batch_sampler,
            }
            (
                self._labeled_dataloader,
                self._unlabeled_dataloader,
            ) = self._train_dataloader.init_dataloader(
                dataset=self._train_dataset,
                sampler=self._train_sampler,
                batch_sampler=self._train_batch_sampler,
                mu=self.mu,
            )

    def start_fit(self, *args, **kwargs):
        self.init_epoch()
        self._network.zero_grad()
        self._network.train()

    def start_fit_epoch(self, *args, **kwargs):
        pass

    def start_fit_batch(self, *args, **kwargs):
        pass

    def train(
            self,
            lb_X=None,
            lb_y=None,
            ulb_X=None,
            lb_idx=None,
            ulb_idx=None,
            *args,
            **kwargs,
    ):
        raise NotImplementedError

    def get_loss(self, train_result, *args, **kwargs):
        raise NotImplementedError

    def optimize(self, loss, *args, **kwargs):
        self._network.zero_grad()
        # print(loss)
        loss.backward()
        self._optimizer.step()
        if self._scheduler is not None:
            self._scheduler.step()
        if self.ema is not None:
            self.ema.update()

    def end_fit_batch(self, train_result, *args, **kwargs):
        self.loss = self.get_loss(train_result)
        self.optimize(self.loss)

    def fit_batch_loop(self, valid_X=None, valid_y=None):
        for (lb_idx, lb_X, lb_y), (ulb_idx, ulb_X, _) in zip(self._labeled_dataloader, self._unlabeled_dataloader):
            if self.it_epoch >= self.num_it_epoch or self.it_total >= self.num_it_total:
                break
            self.start_fit_batch()
            lb_idx = to_device(lb_idx, self.device)
            lb_X = to_device(lb_X, self.device)
            lb_y = to_device(lb_y, self.device)
            ulb_idx = to_device(ulb_idx, self.device)
            ulb_X = to_device(ulb_X, self.device)
            train_result = self.train(lb_X=lb_X, lb_y=lb_y, ulb_X=ulb_X, lb_idx=lb_idx, ulb_idx=ulb_idx)
            self.end_fit_batch(train_result)
            self.it_total += 1
            self.it_epoch += 1
            if self.verbose:
                print(self.it_total, file=self.file)
                print(self.it_total)
            if valid_X is not None and self.eval_it is not None and self.it_total % self.eval_it == 0:
                self.evaluate(X=self.unlabeled_X, y=self.unlabeled_y, valid=True)
                self.evaluate(X=valid_X, y=valid_y, valid=True)
                self.valid_performance.update(
                    {"epoch_" + str(self._epoch) + "_it_" + str(self.it_epoch): self.performance}
                )

    def end_fit_epoch(self, *args, **kwargs):
        pass

    def fit_epoch_loop(self, valid_X=None, valid_y=None):
        self.valid_performance = {}
        self.it_total = 0
        for self._epoch in range(1, self.epoch + 1):
            self.it_epoch = 0
            if self.it_total >= self.num_it_total:
                break
            self.start_fit_epoch()
            self.fit_batch_loop(valid_X, valid_y)
            self.end_fit_epoch()
            if valid_X is not None and self.eval_epoch is not None and self._epoch % self.eval_epoch == 0:
                self.evaluate(X=valid_X, y=valid_y, valid=True)
                self.valid_performance.update(
                    {"epoch_" + str(self._epoch) + "_it_" + str(self.it_epoch): self.performance}
                )

        if valid_X is not None and (self.eval_epoch is None or self.epoch % self.eval_epoch != 0):
            self.evaluate(X=valid_X, y=valid_y, valid=True)
            self.valid_performance.update({"epoch_" + str(self._epoch) + "_it_" + str(self.it_epoch): self.performance})

    def end_fit(self, *args, **kwargs):
        pass

    def fit(
            self,
            X=None,
            y=None,
            unlabeled_X=None,
            valid_X=None,
            valid_y=None,
            unlabeled_y=None,
    ):
        self.unlabeled_X = unlabeled_X
        self.unlabeled_y = unlabeled_y
        self.init_train_dataset(X, y, unlabeled_X)
        self.init_train_dataloader()
        if self.network is not None:
            self.init_model()
            self.init_ema()
            self.init_optimizer()
            self.init_scheduler()
        self.init_augmentation()
        self.init_transform()
        self.start_fit()
        self.fit_epoch_loop(valid_X, valid_y)
        self.end_fit()
        return self

    def init_estimate_dataset(self, X=None, valid=False):
        self._valid_dataset = copy.deepcopy(self.valid_dataset)
        self._test_dataset = copy.deepcopy(self.test_dataset)
        if valid:
            if isinstance(X, Dataset):
                self._valid_dataset = X
            else:
                self._valid_dataset = self._valid_dataset.init_dataset(X=X)
        else:
            if isinstance(X, Dataset):
                self._test_dataset = X
            else:
                self._test_dataset = self._test_dataset.init_dataset(X=X)

    def init_estimate_dataloader(self, valid=False):
        self._valid_dataloader = copy.deepcopy(self.valid_dataloader)
        self._test_dataloader = copy.deepcopy(self.test_dataloader)
        self._valid_sampler = copy.deepcopy(self.valid_sampler)
        self._test_sampler = copy.deepcopy(self.test_sampler)
        self._valid_batch_sampler = copy.deepcopy(self.valid_batch_sampler)
        self._test_batch_sampler = copy.deepcopy(self.test_batch_sampler)
        if valid:
            self._estimate_dataloader = self._valid_dataloader.init_dataloader(
                self._valid_dataset,
                sampler=self._valid_sampler,
                batch_sampler=self._valid_batch_sampler,
            )
        else:
            self._estimate_dataloader = self._test_dataloader.init_dataloader(
                self._test_dataset,
                sampler=self._test_sampler,
                batch_sampler=self._test_batch_sampler,
            )

    def start_predict(self, *args, **kwargs):
        self._network.eval()
        if self.ema is not None:
            self.ema.apply_shadow()
        self.y_est = torch.Tensor().to(self.device)

    def start_predict_batch(self, *args, **kwargs):
        pass

    @torch.no_grad()
    def estimate(self, X, idx=None, *args, **kwargs):
        outputs = self._network(X)
        return outputs

    def end_predict_batch(self, *args, **kwargs):
        pass

    def predict_batch_loop(self):
        with torch.no_grad():
            for idx, X, _ in self._estimate_dataloader:
                self.start_predict_batch()
                idx = to_device(idx, self.device)
                X = X[0] if isinstance(X, (list, tuple)) else X
                X = to_device(X, self.device)
                _est = self.estimate(X=X, idx=idx)
                _est = _est[0] if isinstance(_est, (list, tuple)) else _est
                self.y_est = torch.cat((self.y_est, _est), 0)
                self.end_predict_batch()

    @torch.no_grad()
    def get_predict_result(self, y_est, *args, **kwargs):
        if self._estimator_type == "classifier" or "classifier" in self._estimator_type:
            y_score = Softmax(dim=-1)(y_est)
            max_probs, y_pred = torch.max(y_score, dim=-1)
            y_pred = y_pred.cpu().detach().numpy()
            self.y_score = y_score.cpu().detach().numpy()
            return y_pred
        else:
            self.y_score = y_est.cpu().detach().numpy()
            y_pred = self.y_score
            return y_pred

    def end_predict(self, *args, **kwargs):
        self.y_pred = self.get_predict_result(self.y_est)
        if self.ema is not None:
            self.ema.restore()
        self._network.train()

    @torch.no_grad()
    def predict(self, X=None, valid=False):
        self.init_estimate_dataset(X, valid)
        self.init_estimate_dataloader(valid)
        self.start_predict()
        self.predict_batch_loop()
        self.end_predict()
        return self.y_pred

    @torch.no_grad()
    def predict_proba(self, X=None, valid=False):
        self.init_estimate_dataset(X, valid)
        self.init_estimate_dataloader(valid)
        self.start_predict()
        self.predict_batch_loop()
        self.end_predict()
        return self.y_score

    @torch.no_grad()
    def evaluate(self, X, y=None, valid=False):
        if isinstance(X, Dataset) and y is None:
            y = getattr(X, "y")

        self.y_pred = self.predict(X, valid=valid)
        self.y_score = self.y_score

        if self.evaluation is None:
            return None
        elif isinstance(self.evaluation, (list, tuple)):
            performance = []
            for eval in self.evaluation:
                if self._estimator_type == "classifier" or "classifier" in self._estimator_type:
                    score = eval.scoring(y, self.y_pred, self.y_score)
                else:
                    score = eval.scoring(y, self.y_pred)
                performance.append(score)
                if self.verbose:
                    print(score, file=self.file)
                    print(score)
            self.performance = performance
            return performance
        elif isinstance(self.evaluation, dict):
            performance = {}
            for key, val in self.evaluation.items():
                if self._estimator_type == "classifier" or "classifier" in self._estimator_type:
                    performance[key] = val.scoring(y, self.y_pred, self.y_score)
                else:
                    performance[key] = val.scoring(y, self.y_pred)
                if self.verbose:
                    print(key, " ", performance[key], file=self.file)
                    print(key, " ", performance[key])
                self.performance = performance
            return performance
        else:
            if self._estimator_type == "classifier" or "classifier" in self._estimator_type:
                performance = self.evaluation.scoring(y, self.y_pred, self.y_score)
            else:
                performance = self.evaluation.scoring(y, self.y_pred)
            if self.verbose:
                print(performance, file=self.file)
                print(performance)
            self.performance = performance
            return performance


class DeepModelMixinCIL:
    batch_size = 64
    EPSILON = 1e-8

    def __init__(
            self,
            network: None,
            memory_size: int = 0,
            memory_per_class: int = 0,
            fixed_memory: bool = False,
            device: int = 0,
            multiple_gpus: list = [0],
            seed: int = 1989,
            evaluation_topk: int = 5,
            evaluation_period: int = 1,
    ):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        self._network = network
        self._old_network = None

        # evaluation
        self.topk = evaluation_topk
        self.eval_period = evaluation_period

        # memory
        self._memory_size = memory_size
        self._memory_per_class = memory_per_class
        self._fixed_memory = fixed_memory

        # device
        self._device = device
        self._multiple_gpus = multiple_gpus

        self._set_random(seed)

        self.train_loader = None
        self.test_loader = None

        logs_name = "logs/CIL_CIFAR100/"

        if not os.path.exists(logs_name):
            os.makedirs(logs_name)

        # TODO: lack of flexibility
        logfilename = "logs/CIL_CIFAR100/ResNet_" + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        logging.info("Log file: {}".format(logfilename))
        print("Log file: {}".format(logfilename))

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[
                logging.FileHandler(filename=logfilename + ".log"),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def fit(self, dataset: CILDataset):

        cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}

        for task_id in range(dataset.nb_tasks):
            self.incremental_train(dataset)
            cnn_accy, nme_accy = self.eval_task(dataset)
            self.after_task()

            if nme_accy is not None:
                logging.info("CNN: {}".format(cnn_accy["grouped"]))
                logging.info("NME: {}".format(nme_accy["grouped"]))

                cnn_curve["top1"].append(cnn_accy["top1"])
                cnn_curve["top5"].append(cnn_accy["top5"])

                nme_curve["top1"].append(nme_accy["top1"])
                nme_curve["top5"].append(nme_accy["top5"])

                print("CNN top1 curve: {}".format(cnn_curve["top1"]))
                print("NME top1 curve: {}".format(nme_curve["top1"]))

                logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
                logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
                logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
                logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

                print('Average Accuracy (CNN):', np.around(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), decimals=2))
                print('Average Accuracy (NME):', np.around(sum(nme_curve["top1"]) / len(nme_curve["top1"]), decimals=2))

                logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
                logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"]) / len(nme_curve["top1"])))
            else:
                logging.info("No NME accuracy.")
                logging.info("CNN: {}".format(cnn_accy["grouped"]))

                cnn_curve["top1"].append(cnn_accy["top1"])
                cnn_curve["top5"].append(cnn_accy["top5"])

                print("CNN top1 curve: {}".format(cnn_curve["top1"]))

                logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
                logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

                print('Average Accuracy (CNN):', np.around(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), decimals=2))
                logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))

    def incremental_train(self, dataset: CILDataset):
        raise NotImplementedError

    def eval_task(self, dataset: CILDataset) -> (dict, dict):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + self.EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def after_task(self):
        self._known_classes = self._total_classes

    @torch.no_grad()
    def predict(self, X: np.ndarray, cls_mode: Literal['fc', 'nme'] = 'fc', topk=1) -> np.ndarray:
        """
        :param cls_mode: str, either 'fc' or 'nme'; type of classifiers
        """
        self._network.eval()
        X = X.reshape(-1, 3, 32, 32)
        X = torch.from_numpy(X).to(self._device).float()

        if cls_mode.lower() == 'fc':
            outputs = self._network(X)["logits"]
            preds = torch.topk(outputs, k=topk, dim=1, largest=True, sorted=True)[1]
            preds = np.array(preds.cpu())
        else:
            if isinstance(self._network, nn.DataParallel):
                vectors = tensor2numpy(
                    self._network.module.extract_vector(X)
                )
            else:
                vectors = tensor2numpy(
                    self._network.extract_vector(X)
                )
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + self.EPSILON)).T

            dists = cdist(self._class_means, vectors, "sqeuclidean")
            preds = np.argsort(dists.T, axis=1)[:, : topk]

        return preds

    @torch.no_grad()
    def evaluate(self, y_pred, y_true):
        return self._evaluate(y_pred, y_true)

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = self.accuracy(y_pred.T[0], y_true, self._known_classes, self.eval_period)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    @staticmethod
    def accuracy(y_pred: np.ndarray, y_true: np.array, nb_old, increment=1):
        assert len(y_pred) == len(y_true), "Data length error."
        all_acc = {}
        all_acc["total"] = np.around(
            (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
        )

        # Grouped accuracy
        for class_id in range(0, np.max(y_true), increment):
            idxes = np.where(
                np.logical_and(y_true >= class_id, y_true < class_id + increment)
            )[0]
            label = "{}-{}".format(
                str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
            )
            all_acc[label] = np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )

        # Old accuracy
        idxes = np.where(y_true < nb_old)[0]
        all_acc["old"] = (
            0
            if len(idxes) == 0
            else np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )
        )

        # New accuracy
        idxes = np.where(y_true >= nb_old)[0]
        all_acc["new"] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

        return all_acc

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=self.batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + self.EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=self.batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + self.EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=self.batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + self.EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + self.EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + self.EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + self.EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    def _set_device(self):
        device_type = self._multiple_gpus
        gpus = []

        for device in device_type:
            if device_type == -1:
                device = torch.device("cpu")
            else:
                device = torch.device("cuda:{}".format(device))

            gpus.append(device)

        self._multiple_gpus = gpus

    def _set_random(self, seed):
        print("Setting random seed to {}".format(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        np.random.seed(seed)
