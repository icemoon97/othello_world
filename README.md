# Exploring the Limits of OthelloGPT's Emergent Representations

## Overview

This repo contains code and data for my investigation of linear representations in OthelloGPT, a small transformer model trained to predict legal moves in the game Othello. This project follows up on the work by [Li et al.](https://arxiv.org/abs/2210.13382) and [Neel Nanda](https://www.lesswrong.com/posts/nmxzr2zsjNtjaHh7x/actually-othello-gpt-has-a-linear-emergent-world).

The project investigates the emergent linear representations within transformer models, demonstrating that targeted interventions using directions learned by linear probes can achieve near-perfect edit performance. It also introduces a novel technique for "global intervention," where a full board state is substituted into the residual stream during inference. Linear probes are also used to extract and edit representations beyond the board state, with interventions on the next turn color coherently altering the model's board representation and outputs.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Generating Synthetic Data](#generating-synthetic-data)
  - [Training Othello-GPT](#training-othello-gpt)
  - [Probing Othello-GPT](#probing-othello-gpt)
  - [Interventions](#interventions)
- [Acknowledgements](#acknowledgements)

## Installation

```bash
conda env create -f environment.yml
conda activate othello
python -m ipykernel install --user --name othello --display-name "Othello"
```

## Usage
### Generating Synthetic Data
To use the original paper's datasets: download the [championship dataset](https://drive.google.com/drive/folders/1KFtP7gfrjmaoCV-WFC4XrdVeOxy1KmXe?usp=sharing) and the [synthetic dataset](https://drive.google.com/drive/folders/1pDMdMrnxMRiDnUd-CNfRNvZCi7VXFRtv?usp=sharing) and save them in `data` subfolder.

To generate new synthetic dataset, see `data/othello.py`. It contains code to generate synthetic games with a 'playertype' component that is biased in a particular cardinal direction (details in full report).

### Training Othello-GPT
See `train_othello_model.ipynb`. For this project, training used 4 GPUs and took roughly 8 hours.

### Probing Othello-GPT
See `train_probe_othello.py`. It contains flags to train probes for board state, current turn, or playertype (only valid on model trained with synthetic playertype data).

### Interventions
See `interventions.ipynb` for Li et al's original intervention method which used non-linear probes and gradient descent.

See `linear_board_probes.ipynb` for interventions with linear probes, including 'global' intervention and interventions on next turn.

See `probing_playertype.ipynb` for intervention experiments using synthetic dataset with playertype.


## Acknowledgements

I am deeply thankful to Kenneth Li and his colleagues for their pioneering work on OthelloGPT, and for making their code and datasets public. This project was initially forked from their repository, which can be [found here](https://github.com/likenneth/othello_world).
