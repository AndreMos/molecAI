#! /usr/bin/env python3
#SBATCH -p gpu
#SBATCH -D /data/scratch/andrem97
#SBATCH --gres=gpu:1
#SBATCH --time 10:30:00
#SBATCH -J testjob
#SBATCH --mem 4GB
import sys
import os
sys.path.append(os.getcwd())
# -*- coding: utf-8 -*-
"""Untitled9.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1de_AxhXMN_HcBwbXSUkwUu-W0eNLJGg9
"""

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from pathlib import Path
# from joblib import Parallel, delayed
import pytorch_lightning  as pl# import LightningModule, Trainer
import torch_geometric
from model import Schnet




# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
#


from torch_geometric.datasets import QM9
dataset = QM9('/data/scratch/andrem97/', pre_transform=torch_geometric.transforms.Distance(norm=False,cat=False))


class DataModule(pl.LightningDataModule):

  def train_dataloader(self):
    return torch_geometric.loader.DataLoader(dataset[:110462], shuffle=True, batch_size = 32, **{'drop_last' : True})

  def val_dataloader(self):
    return torch_geometric.loader.DataLoader(dataset[110462:111462], shuffle=True, batch_size = 32, **{'drop_last' : True})

data_module = DataModule()

# train
model = Schnet()
trainer = pl.Trainer()

# Commented out IPython magic to ensure Python compatibility.
# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/

trainer.fit(model, data_module)



