#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import argparse
import torch
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED,seed_worker
setup_seed(SEED)
g = torch.Generator()
g.manual_seed(SEED)
########### Import your packages below ##########
from data.dataset import E2EDataset, VOCAB
from trainer import TrainConfig


def parse():
    parser = argparse.ArgumentParser(description='training')
    # data
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, required=True, help='path to valid set')

    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='exponential decay from lr to final_lr')
    parser.add_argument('--warmup', type=int, default=0, help='linear learning rate warmup')
    parser.add_argument('--max_epoch', type=int, default=10, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model and logs')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--patience', type=int, default=1000, help='patience before early stopping (set with a large number to turn off early stopping)')
    parser.add_argument('--save_topk', type=int, default=10, help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=4)

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    # model
    parser.add_argument('--model_type', type=str, required=True, choices=['RNAModel','RNAmaskModel','RNAGNNModel'],
                        help='Type of model')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of residue/atom embedding')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of hidden states')
    parser.add_argument('--k_neighbors', type=int, default=9, help='Number of neighbors in KNN graph')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    # task setting
 


    parser.add_argument('--fix_channel_weights', action='store_true', help='Fix channel weights, may also for special use (e.g. antigen with modified AAs)')
 

    return parser.parse_args()


def main(args):
    torch.autograd.set_detect_anomaly(True)
    ########### load your train / valid set ###########
    if (len(args.gpus) > 1 and int(os.environ['LOCAL_RANK']) == 0) or len(args.gpus) == 1:
        print_log(args)
 

    train_set = E2EDataset(args.train_set)
    valid_set = E2EDataset(args.valid_set)


    ########## set your collate_fn ##########
    collate_fn = train_set.collate_fn

    ########## define your model/trainer/trainconfig #########
    config = TrainConfig(**vars(args))

    if args.model_type == 'RNAModel':
        from trainer import RNAmaskModelTrainer as Trainer
        from model import RNAModel
        model = RNAModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type()+1, VOCAB.get_mask_idx(),
                   args.k_neighbors,n_layers=args.n_layers,)

    elif args.model_type == 'RNAmaskModel':
        from trainer import RNAmaskModelTrainer as Trainer
        from model import RNAmaskModel
        model = RNAmaskModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type()+1, VOCAB.get_mask_idx(),
                   args.k_neighbors,n_layers=args.n_layers)   

    elif args.model_type == 'RNAGNNModel':
        from trainer import RNAmaskModelTrainer as Trainer
        from model import RNAGNNModel
        model = RNAGNNModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type()+1, VOCAB.get_mask_idx(),
                   args.k_neighbors,n_layers=args.n_layers)  


    else:
        raise NotImplemented(f'model {args.model_type} not implemented')

    step_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    config.add_parameter(step_per_epoch=step_per_epoch)

    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=args.shuffle)
        args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        args.local_rank = -1
        train_sampler = None
    config.local_rank = args.local_rank

    if args.local_rank == 0 or args.local_rank == -1:
        print_log(f'step per epoch: {step_per_epoch}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=collate_fn,
                              worker_init_fn=seed_worker,
                              generator=g)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn,
                              worker_init_fn=seed_worker,
                              generator=g)
    
    trainer = Trainer(model, train_loader, valid_loader, config)
    trainer.train(args.gpus, args.local_rank)


if __name__ == '__main__':
    args = parse()
    main(args)


