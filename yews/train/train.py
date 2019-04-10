import numpy as np
import torch
from torch import optim

from . import functional as F

class Trainer(object):

    def __init__(self, model_gen, criterion, lr=0.1):
        # default device
        self.device = F.get_torch_device()

        # model
        self.model_gen = model_gen
        self.model = None

        # optimizer
        self.criterion = criterion
        self.lr = lr
        self.optimizer = None
        self.scheduler = None

        # training process
        self.start_epoch = 0
        self.end_epoch = None

        # results
        self.best_acc = None
        self.best_model = None
        self.train_loss = None
        self.train_acc = None
        self.val_loss = None
        self.val_acc = None

        # reset
        self.reset()


    def reset(self):
        self.model = self.model_gen()
        self.model = F.model_on_device(self.model, self.device)

        self.reset_optimizer()
        self.reset_scheduler()

        self._reset_results()

    def reset_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def reset_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              verbose=True)

    def _update_scheduler(self):
        self.scheduler.step(self.val_loss[-1])


    def _reset_results(self):
        self.best_acc = 0.
        self.best_model = None
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def update_train_results(self, acc, loss):
        self.train_acc.append(acc)
        self.train_loss.append(loss)

    def update_val_results(self, acc, loss):
        self.val_acc.append(acc)
        self.val_loss.append(loss)

    def validate(self, loader, print_freq=None):
        return F.validate(self.model, loader, self.criterion, print_freq=print_freq)

    def train_one_epoch(self, loader, epoch, print_freq=None):
        return F.train(self.model, loader, self.criterion, self.optimizer,
                       epoch, print_freq=print_freq)

    def train(self, train_loader, val_loader, epochs=100, print_freq=None):

        start_epoch = 0
        end_epoch = epochs

        # record results for initial model
        print("Validation on training set.")
        acc, loss = self.validate(train_loader)
        self.update_train_results(acc, loss)
        print("Validation on valiation set.")
        acc, loss = self.validate(val_loader)
        self.update_val_results(acc, loss)

        # training loop
        print("Start training ...")
        for epoch in range(start_epoch, end_epoch):
            # update learning rate
            self._update_scheduler()

            # train model for one epoch
            acc, loss = self.train_one_epoch(train_loader, epoch, print_freq=print_freq)
            self.update_train_results(acc, loss)

            # validate model for current epoch
            acc, loss = self.validate(val_loader)
            self.update_val_results(acc, loss)

            # preserve best model and accuracy
            is_best = self.val_acc[-1] > self.best_acc
            self.best_acc = max(self.val_acc[-1], self.best_acc)
        print(f"Training fisihed. Best accuracy is {self.best_acc}")

