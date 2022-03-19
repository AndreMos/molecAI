#! /usr/bin/env python3
# fmt: off
# SBATCH -o distilbertONcvconv
# SBATCH -p gpu
# SBATCH -D /data/scratch/andrem97
# SBATCH --gres=gpu:1
# SBATCH --time 16:00:00
# SBATCH -J testjob
# SBATCH --mem 4GB
# fmt: on
import sys
import os


sys.path.append(os.getcwd())


from models.model_bert import DistilBertAppl
import torch_geometric
import pytorch_lightning as pl

# from model_mult_mod import  MultiMod
# from model_perceiver import MyPerceiver, MolecPreprocessor


from datasets.dataset_mm import CustomDataset


# DO NOT FORGET TO CUT PRETRANSFORM
dataset = CustomDataset(
    "/data/scratch/andrem97/",
    pre_transform=torch_geometric.transforms.Distance(norm=False, cat=False),
)


class DataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return torch_geometric.loader.DataLoader(
            dataset[:110462], shuffle=True, batch_size=32, **{"drop_last": True}
        )

    def val_dataloader(self):
        return torch_geometric.loader.DataLoader(
            dataset[110462:111462], batch_size=32, **{"drop_last": True}
        )


data_module = DataModule()

# train

# config = PerceiverConfig(num_latents=64, d_latents=128, num_labels=1, num_cross_attention_heads=4)
# decoder = PerceiverClassificationDecoder(
#     config,
#     num_channels=config.d_latents,
#     trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
#     use_query_residual=True,
# )
#
# model = MyPerceiver(config, input_preprocessor=MolecPreprocessor(), decoder=decoder)


model = DistilBertAppl()

trainer = pl.Trainer(gpus=1)

# Commented out IPython magic to ensure Python compatibility.
# Start tensorboard
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs

trainer.fit(model, data_module)
