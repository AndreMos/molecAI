#! /usr/bin/env python3
#SBATCH -o perceiveronsummedposenc
#SBATCH -p gpu
#SBATCH -D /data/scratch/andrem97
#SBATCH --gres=gpu:1
#SBATCH --time 24:00:00
#SBATCH -J testjob
#SBATCH --mem 8GB
import sys
import os


sys.path.append(os.getcwd())

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
)
from models.model_bert import DistilBertAppl
import torch_geometric
import pytorch_lightning as pl

# from model_mult_mod import  MultiMod
from models.model_perceiver import MyPerceiver, MolecPreprocessor


from datasets.dataset_perceiver import CustomDataset


# DO NOT FORGET TO CUT PRETRANSFORM
dataset = CustomDataset(
    "/data/scratch/andrem97/",
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

config = PerceiverConfig(
    num_latents=128, d_latents=512, num_labels=1,num_cross_attention_heads=2,
    num_self_attends_per_block=5, attention_probs_dropout_prob=0, num_self_attention_heads=2
)
decoder = PerceiverClassificationDecoder(
    config,
    num_channels=config.d_latents,
    trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
    use_query_residual=True,
)

model = MyPerceiver(config, input_preprocessor=MolecPreprocessor(), decoder=decoder)


#model = DistilBertAppl()

trainer = pl.Trainer(gpus=1)

# Commented out IPython magic to ensure Python compatibility.
# Start tensorboard
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs

trainer.fit(model, data_module)
