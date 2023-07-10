#!/bin/bash
for C in bias/TLbias50 bias/TLbias80 bias/TLbias95 bias/TLcontrol
do

for X in {1..8}
do
    echo "$C probe layer $X"
    python3 train_probe_othello.py --layer $X --twolayer --mid_dim 256 --ckpt $C
    # python3 train_probe_othello.py --layer $X --twolayer --mid_dim 256 --random
done

done
