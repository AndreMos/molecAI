#!/usr/bin/env python3
#SBATCH -o perceiverbondswobonds
#SBATCH -p gpu
#SBATCH -D /data/scratch2/andrem97/molecAI/
#SBATCH --gres=gpu:1
#SBATCH -J testjob
#SBATCH --mem 8GB
import sys
import os


sys.path.append(os.getcwd())

import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (
    PerceiverForSequenceClassification,
    PerceiverConfig,
    PerceiverTokenizer,
    PerceiverFeatureExtractor,
    PerceiverModel,
    PerceiverForMultimodalAutoencoding,
    PerceiverForImageClassificationLearned,
)
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverTextPreprocessor,
    PerceiverImagePreprocessor,
    PerceiverClassificationDecoder,
    AbstractPreprocessor,
    PerceiverMultimodalPreprocessor,
)
from models.model_bert import DistilBertAppl
import torch_geometric
import pytorch_lightning as pl
from pathlib import Path

from dagshub.pytorch_lightning import DAGsHubLogger

# from model_mult_mod import  MultiMod
from models.model_perceiver import MyPerceiver, MolecPreprocessor, AnglePreprocessor


from datasets.dataset_perceiver import CustomDataset


from hydra import initialize, initialize_config_module, initialize_config_dir, compose
import codes
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/AndreMos/molecAI.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "AndreMos"
os.environ["MLFLOW_TRACKING_PASSWORD"] = codes.p
mlflow.pytorch.autolog()
mlflow.set_experiment("perceiverbondswithoutbonds")

base_path = Path.cwd()

with initialize_config_dir(config_dir=str(base_path/"configs")):
    cfg = compose(config_name="config.yaml")

# DO NOT FORGET TO CUT PRETRANSFORM
dataset = CustomDataset(
    cfg.dir,
)


class DataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return torch_geometric.loader.DataLoader(
            dataset[:110462],
            shuffle=True,
            batch_size=cfg.perceiver.batch_size,
            **{"drop_last": True}
        )

    def val_dataloader(self):
        return torch_geometric.loader.DataLoader(
            dataset[110462:111462],
            batch_size=cfg.perceiver.batch_size,
            **{"drop_last": True}
        )


# train
data_module = DataModule()
allias = cfg.perceiver.initial
config = PerceiverConfig(
    num_latents=allias.num_latents,
    d_latents=allias.d_latents,
    num_labels=allias.num_labels,
    num_cross_attention_heads=allias.num_cross_attention_heads,
    num_self_attends_per_block=allias.num_self_attends_per_block,
    attention_probs_dropout_prob=allias.attention_probs_dropout_prob,
    num_self_attention_heads=allias.num_self_attention_heads,
)
min_padding_size = cfg.perceiver.min_padding_size


decoder = PerceiverClassificationDecoder(
    config,
    num_channels=config.d_latents,
    trainable_position_encoding_kwargs=dict(
        num_channels=config.d_latents, index_dims=1
    ),
    use_query_residual=True,
)
preprocessor = PerceiverMultimodalPreprocessor(
    min_padding_size=min_padding_size,
    modalities=nn.ModuleDict(
        {"angles": AnglePreprocessor(), "dist": MolecPreprocessor()}
    ),
)

model = MyPerceiver(config, input_preprocessor=preprocessor, decoder=decoder)
#model.save_hyperparameters()
#with mlflow.start_run():
trainer = pl.Trainer(gpus=1, callbacks=[EarlyStopping(monitor="val_loss", patience=20, min_delta=5e-2)])
trainer.fit(model, data_module)
#mlflow.end_run()


