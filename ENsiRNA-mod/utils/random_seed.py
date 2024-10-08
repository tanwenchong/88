#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import numpy as np
import random
import os

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
     torch.use_deterministic_algorithms(True)
     
def seed_worker(worker_id):
    random.seed(SEED + worker_id)

SEED = 12
