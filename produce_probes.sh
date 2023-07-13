#!/bin/bash

for X in {1..8}
do
    echo "$C probe layer $X"
    # python3 train_probe_othello.py --layer $X --twolayer --mid_dim 256 --ckpt $C
    python3 train_probe_othello.py --layer $X --twolayer --mid_dim 256 --random
done
