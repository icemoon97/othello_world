#!/bin/bash

for X in 1 2 3 4 5 6 7 8
do
    echo "$C probe layer $X"
    python3 train_probe_othello.py --layer $X --ckpt playertype/playertype_e40 --type turn
    # python3 train_probe_othello.py --layer $X --twolayer --mid_dim 256 --random
done
