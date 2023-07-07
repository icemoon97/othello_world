# make deterministic
from mingpt.utils import set_seed
set_seed(44)

import os
import math
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from data.othello import Othello, OthelloBoardState, permit_reverse
from mingpt.dataset import CharDataset
from mingpt.utils import sample
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train GPT model on Othello dataset')
    
    parser.add_argument('--dataroot',
                    required=True,
                    type=str)
    parser.add_argument('--exp_name',
                    default="",
                    type=str)
    
    args, _ = parser.parse_known_args()
    

    # n_games=-1 means use as many simulated games as possible (from "data/othello_synthetic/")
    othello = Othello(n_games=-1, data_root=args.dataroot)
    train_dataset = CharDataset(othello)
    # original OthelloGPT params: n_layer=8, n_head=8, n_embd=512
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = model.to(device)
    else:
        print("NO GPU FOUND")

    # setting up training
    max_epochs = 40
    experiment_name = args.exp_name
    t_start = time.strftime("_%Y%m%d_%H%M%S")
    ckpt_path = f"./ckpts/{experiment_name}_{t_start}.ckpt"
    tconf = TrainerConfig(
        max_epochs=max_epochs, 
        batch_size=512*4, # using 4 gpus
        learning_rate=5e-4,
        lr_decay=True, 
        warmup_tokens=len(train_dataset)*train_dataset.block_size*5, 
        final_tokens=len(train_dataset)*train_dataset.block_size*max_epochs,
        num_workers=0, 
        ckpt_path=ckpt_path, 
        # saved_epochs=[0, 1, 2, 3, 5, 10, 15, 20],
    )
    trainer = Trainer(model, train_dataset, None, tconf)
    device = trainer.device
    print(t_start)

    trainer.train()