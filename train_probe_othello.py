import os

# make deterministic
from mingpt.utils import set_seed
set_seed(42)

import time
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from data.othello import Othello, OthelloBoardState
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig, GPTforProbing
from mingpt.probe_trainer import Trainer, TrainerConfig
from mingpt.probe_model import BatteryProbeClassification, BatteryProbeClassificationTwoLayer

parser = argparse.ArgumentParser(description='Train classification network')
parser.add_argument('--layer',
                    required=True,
                    default=-1,
                    type=int)

parser.add_argument('--epo',
                    default=16,
                    type=int)

parser.add_argument('--twolayer',
                    dest='twolayer', 
                    action='store_true')

parser.add_argument('--mid_dim',
                    default=128,
                    type=int)

# means testing against randomly initialized model rather than a ckpt
parser.add_argument('--random',
                    dest='random', 
                    action='store_true')

parser.add_argument('--ckpt',
                    dest='ckpt', 
                    type=str)

# should be "state" for board state probe
# or "player" for player type probe
# or "turn" for turn probe
parser.add_argument('--type',
                    default="state", 
                    type=str)

args, _ = parser.parse_known_args()

folder_name = f"playertype/probes/{args.type}"

if args.twolayer:
    folder_name = folder_name + f"_tl{args.mid_dim}"  # tl for probes without batchnorm
if args.random:
    folder_name = folder_name + "_random"

print(f"Running experiment for {folder_name}")
othello = Othello(data_root="othello_1player", n_games=10000, deduplicate=False, test_split=0)

player_types, games = zip(*othello)
train_dataset = CharDataset(games)
# train_dataset = CharDataset(othello)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPTforProbing(mconf, probe_layer=args.layer)
if args.random:
    model.apply(model._init_weights)
if args.ckpt:  
    load_res = model.load_state_dict(torch.load(f"./ckpts/{args.ckpt}.ckpt"))
else:
    raise Exception("not given ckpt path or random flag")

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = model.to(device)

# creating dataset of activations and properties
loader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=1, num_workers=1)
act_container = []
property_container = []

for i, (x, y) in tqdm(enumerate(loader), total=len(loader)):
    tbf = [train_dataset.itos[_] for _ in x.tolist()[0]]
    # truncates game if it is less than 60 moves
    valid_until = tbf.index(-100) if -100 in tbf else 999

    properties = []
    # get properties (board state or player type)
    if args.type == "state":
        ob = OthelloBoardState()
        for i, move in enumerate(tbf[:valid_until]):
            ob.update([move])

            # flipping states so always from moving player's perspective
            fixed_state = np.array(ob.get_state())
            if ob.get_next_hand_color() == 1:
                fixed_state = 2 - fixed_state
            properties.append(fixed_state)

    # elif args.type == "player":
    #     properties = [[player_types[i]] for _ in range(len(tbf[:valid_until]))]
    
    elif args.type == "turn":
        ob = OthelloBoardState()
        for i, move in enumerate(tbf[:valid_until]):
            ob.update([move])

            properties.append([ob.get_next_hand_color()])

    property_container.extend(properties)

    # gets activations for each move
    act = model(x.to(device))[0, ...].detach().cpu()  # [block_size, f]
    act = np.array([_[0] for _ in act.split(1, dim=0)[:valid_until]])
    act_container.extend(act)

    assert len(act_container) == len(property_container)

# creating probe
if args.type == "state":
    probe_class = 3
    num_task = 64
elif args.type == "player":
    probe_class = 4
    num_task = 1
elif args.type == "turn":
    probe_class = 2
    num_task = 1
else:
    raise Exception("invalid probe type given")

if args.twolayer:
    probe = BatteryProbeClassificationTwoLayer(device, probe_class=probe_class, num_task=num_task, mid_dim=args.mid_dim)
else:
    probe = BatteryProbeClassification(device, probe_class=probe_class, num_task=num_task)
    

class ProbingDataset(Dataset):
    def __init__(self, act, y):
        assert len(act) == len(y)
        print(f"{len(act)} pairs loaded...")
        self.act = act
        self.y = y
        # print(np.sum(np.array(y)==0), np.sum(np.array(y)==1), np.sum(np.array(y)==2))
        print("y:", np.unique(y, return_counts=True))
        
    def __len__(self, ):
        return len(self.y)
    def __getitem__(self, idx):
        return self.act[idx], torch.tensor(self.y[idx]).to(torch.long)

probing_dataset = ProbingDataset(act_container, property_container)
train_size = int(0.8 * len(probing_dataset))
test_size = len(probing_dataset) - train_size
probe_train_dataset, probe_test_dataset = torch.utils.data.random_split(probing_dataset, [train_size, test_size])


max_epochs = args.epo
t_start = time.strftime("_%Y%m%d_%H%M%S")
tconf = TrainerConfig(
    max_epochs=max_epochs, 
    batch_size=1024, 
    learning_rate=1e-3,
    betas=(.9, .999), 
    lr_decay=True, 
    warmup_tokens=len(train_dataset)*5, 
    final_tokens=len(train_dataset)*max_epochs,
    num_workers=0, 
    weight_decay=0., 
    ckpt_path=os.path.join("./ckpts/", folder_name, f"layer{args.layer}")
)
trainer = Trainer(probe, probe_train_dataset, probe_test_dataset, tconf)
trainer.train(prt=True)
trainer.save_traces()
trainer.save_checkpoint()
