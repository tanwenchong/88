#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import cos, pi, log, exp
import torch
from .abs_trainer import Trainer


class RNACNNModelTrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        self.global_step = 0
        self.epoch = 0
        self.max_step = config.max_epoch * config.step_per_epoch
        self.log_alpha = log(config.final_lr / config.lr) / self.max_step
        self.MLM = False
        super().__init__(model, train_loader, valid_loader, config)

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def get_scheduler(self, optimizer):
        log_alpha = self.log_alpha
        lr_lambda = lambda step: exp(log_alpha * (step + 1))  # equal to alpha^{step}
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=20, min_lr=0.0001)
        return {
            'scheduler': scheduler,
            'frequency': 'batch'
        }

    def train_step(self, batch, batch_idx):
        #batch['context_ratio'] = self.get_context_ratio()
        batch['mode'] = 'Train'
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        #batch['context_ratio'] = 0
        batch['mode'] = 'Valid'
        return self.share_step(batch, batch_idx, val=True)

    ########## Override end ##########



    def share_step(self, batch, batch_idx, val=False):
        #loss, seq_detail, structure_detail, dock_detail, pdev_detail,cos_loss= self.model(**batch)
        if self.MLM == True:
            loss,class_loss,snll= self.model(**batch)
        else:
            loss = self.model(**batch)
       
  

        log_type = 'Validation' if val else 'Train'

        self.log(f'Overall/Loss/{log_type}', loss, batch_idx, val)
        if self.MLM == True:

            self.log(f'Overall/Class_loss/{log_type}', class_loss, batch_idx, val)
            self.log(f'Overall/SNLL_loss/{log_type}', snll, batch_idx, val)


        if not val:
            #lr = self.config.lr if self.scheduler is None else self.scheduler.get_last_lr()
            #lr = lr[0]
            lr = self.config.lr if self.scheduler is None else self.scheduler.optimizer.param_groups[0]['lr']
        
        return loss
