import torch.nn as nn
import torch
import torch_geometric
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_scatter import scatter
import pytorch_lightning as pl
import numpy as np


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


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians, device='cuda:0')
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class MolecPreprocessor(AbstractPreprocessor):
    def __init__(self):
        super().__init__()
        self.rbf_layer = GaussianSmearing()
        ##zero index is USED!!!! 100 might be overkill
        self.emb = nn.Embedding(num_embeddings=100, embedding_dim=128, padding_idx=0)
        # self.to_standart_form = to_standart_form

    def to_standart_form(self, batch_size, param, sample):
        return sample[param].reshape(batch_size, -1)

    @property
    def num_channels(self) -> int:
        return 50 + 128

    def forward(self, sample):
        batch_size = len(sample.idx)
        distance = self.to_standart_form(
            sample=sample, param="distances_padded", batch_size=batch_size
        )
        input_wo_pos = self.rbf_layer(distance).reshape(batch_size, -1, 50)
        # do not forget about batches!!!

        row = self.to_standart_form(
            sample=sample, param="row_padded", batch_size=batch_size
        )
        col = self.to_standart_form(
            sample=sample, param="col_padded", batch_size=batch_size
        )
        pos_enc_row = self.emb(row)
        pos_enc_col = self.emb(col)
        pos_enc = pos_enc_row + pos_enc_col#torch.cat((pos_enc_row, pos_enc_col), dim=-1)
        input_w_pos = torch.cat((input_wo_pos, pos_enc), dim=-1)
        return input_w_pos, None, input_wo_pos

    # def to_standart_form(self, batch_size, param, sample):
    #   return sample[param].reshape(batch_size, -1)


class MyPerceiver(PerceiverModel, pl.LightningModule):
    def forward(
        self,
        inputs,
        attention_mask=None,
        subsampled_output_points=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        res = super().forward(
            # self,
            inputs,
            attention_mask=inputs.attent_dist.reshape(len(inputs.idx), -1),
            subsampled_output_points=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        return res

    def mse(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    def training_step(self, train_batch, batch_idx):
        logits = self.forward(train_batch).logits.reshape(-1)
        loss = self.mse(logits, train_batch.y[:, 7])

        self.log("train_loss", loss, batch_size=32)
        return loss

    def validation_step(self, val_batch, batch_idx):
        logits = self.forward(val_batch).logits.reshape(-1)
        loss = self.mse(logits, val_batch.y[:, 7])
        self.log("val_loss", loss, batch_size=32)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        sched = torch.optim.lr_scheduler.StepLR(optimizer, 100000,
                                                 0.96)  # torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        return [optimizer], [sched]
