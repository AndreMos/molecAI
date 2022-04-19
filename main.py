#!/usr/bin/env python3
#SBATCH -o perceiverbonds
#SBATCH -p gpu
#SBATCH -D /data/scratch2/andrem97/molecAI/
#SBATCH --gres=gpu:1
#SBATCH --time 48:00:00
#SBATCH -J testjob
#SBATCH --mem 8GB
import sys
import os


sys.path.append(os.getcwd())

import torch.nn as nn
import hydra
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
    PerceiverMultimodalPreprocessor
)
from models.model_bert import DistilBertAppl
import torch_geometric
import pytorch_lightning as pl

# from model_mult_mod import  MultiMod
from models.model_perceiver import MyPerceiver, MolecPreprocessor, AnglePreprocessor


from datasets.dataset_perceiver import CustomDataset



# train
@hydra.main(config_path=base_path + "configs", config_name="config")
def run_pipe(config):
    base_path = "/data/scratch2/andrem97/molecAI/"
    # DO NOT FORGET TO CUT PRETRANSFORM
    dataset = CustomDataset(
        base_path,
    )

    class DataModule(pl.LightningDataModule):
        def train_dataloader(self):
            return torch_geometric.loader.DataLoader(
                dataset[:110462], shuffle=True, batch_size=config.perceiver.batch_size, **{"drop_last": True}
            )


        def val_dataloader(self):
            return torch_geometric.loader.DataLoader(
                dataset[110462:111462], batch_size=config.perceiver.batch_size, **{"drop_last": True}
            )

    data_module = DataModule()

    allias = config.perceiver.initial
    perc_conf = PerceiverConfig(
    num_latents=allias.num_latents, d_latents=allias.d_latents, num_labels=allias.num_labels, num_cross_attention_heads=allias.num_cross_attention_heads, num_self_attends_per_block = allias.num_self_attends_per_block,
     attention_probs_dropout_prob=allias.attention_probs_dropout_prob, num_self_attention_heads=allias.num_self_attention_heads
    )
    min_padding_size = config.perceiver.min_padding_size
    config = perc_conf
    decoder = PerceiverClassificationDecoder(
        config,
        num_channels=config.d_latents,
        trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
        use_query_residual=True,
    )
    preprocessor = PerceiverMultimodalPreprocessor(min_padding_size=min_padding_size,
                             modalities=nn.ModuleDict({
                                 'angles': AnglePreprocessor(),
                                 'dist' : MolecPreprocessor()
                                 })
                                      )

    model = MyPerceiver(config, input_preprocessor=preprocessor, decoder=decoder)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, data_module)

if __name__ =='__main__':
    run_pipe()
