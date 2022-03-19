import torch.nn as nn
import torch

from torch_geometric.nn import MessagePassing

import pytorch_lightning as pl
import numpy as np
from transformers import DistilBertConfig, DistilBertForSequenceClassification


import torch.nn.functional as F

# from torch.nn import Embedding, Linear, ModuleList, Sequential
from math import pi as PI


class DistilBertAppl(pl.LightningModule):
    def __init__(self, batch_size=32, hidden_s=128):
        super(DistilBertAppl, self).__init__()
        self.hidden_s = hidden_s
        self.num_filters = 128
        self.num_gaussians = 50
        self.cutoff = 10
        self.config = DistilBertConfig(
            vocab_size=6,
            max_position_embeddings=29,
            dim=self.hidden_s,
            num_labels=1,
            n_heads=4,
            **{"problem_type": "regression"}
        )
        self.bert = DistilBertForSequenceClassification(self.config)
        self.emb = nn.Embedding(
            num_embeddings=6, embedding_dim=self.hidden_s, padding_idx=0
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.num_gaussians, self.num_filters),
            ShiftedSoftplus(),
            nn.Linear(self.num_filters, self.num_filters),
        )

        self.batch_size = batch_size
        self.cvconf = CFConv(
            self.hidden_s, self.hidden_s, self.num_filters, self.mlp, self.cutoff
        )
        self.expand = GaussianSmearing()

    def forward(self, sample):

        x = self.emb(sample.modif_z)
        # x = self.conv(x, sample.edge_attr.reshape(-1).float(), sample.edge_index)

        x = self.cvconf(
            x,
            sample.edge_index,
            sample.edge_attr.reshape(-1).float(),
            self.expand(sample.edge_attr),
        )

        res = self.bert(
            inputs_embeds=x.reshape(self.batch_size, -1, self.hidden_s),
            attention_mask=sample.attent_mask.reshape(self.batch_size, -1),
            labels=sample.y[:, 7],
        )
        return res["loss"]

    def training_step(self, train_batch, batch_idx):
        loss = self.forward(train_batch)
        self.log("train_loss", loss, batch_size=32)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.forward(val_batch)
        self.log("val_loss", loss, batch_size=32)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        # sched = torch.optim.lr_scheduler.StepLR(optimizer, 100000,
        #                                         0.96)  # torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        return [optimizer]  # , [sched]


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, mlp, cutoff):
        super().__init__(aggr="add")
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.nn = mlp
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):

        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)

        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


# class CosineCutoff(pl.LightningModule):
#     r"""Class of Behler cosine cutoff.
#     .. math::
#        f(r) = \begin{cases}
#         0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
#           & r < r_\text{cutoff} \\
#         0 & r \geqslant r_\text{cutoff} \\
#         \end{cases}
#     Args:
#         cutoff (float, optional): cutoff radius.
#     """
#
#     def __init__(self, cutoff=5.0):
#         super(CosineCutoff, self).__init__()
#         self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
#
#     def forward(self, distances):
#         """Compute cutoff.
#         Args:
#             distances (torch.Tensor): values of interatomic distances.
#         Returns:
#             torch.Tensor: values of cutoff function.
#         """
#         # Compute values of cutoff function
#         cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
#         # Remove contributions beyond the cutoff radius
#         cutoffs *= (distances < self.cutoff).float()
#         return cutoffs
#
#
# class ContConv(MessagePassing, pl.LightningModule):
#     r"""
#     Filtering network
#     """
#
#     def __init__(self):
#         super(ContConv, self).__init__()
#         self.rbf = RbfExpand()
#         self.dense = nn.Sequential(nn.Linear(300, 128), nn.Linear(128, 128))
#         self.cutoff = CosineCutoff()
#
#     def forward(
#         self, x: torch.Tensor, r: torch.Tensor, edge_index: torch.Tensor
#     ) -> torch.Tensor:
#         r"""Forward pass through the Conv block.
#         Parameters
#         ----------
#         x:
#             Embedded properties of charges of the system (n-atoms, n_hidden)
#         r:
#             interatomic distances (n_edges, )
#
#
#         """
#         # print('R do', r.shape)
#         C = self.cutoff(r)
#         r = self.rbf(r)
#         # print('R posle', r.shape)
#
#         for dense_instance in self.dense:
#             r = dense_instance(r)
#             r = F.tanh(r)  # torch.log(0.5 * torch.exp(r) + 0.5)
#         r = r * C.unsqueeze(-1)
#         prop = self.propagate(edge_index, x=x, W=r, size=None)
#         # print(r.shape, x.shape, prop.shape)
#         return prop  # x * r.view(-1, 1)
#
#     def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
#         return x_j * W
#
#
# class RbfExpand(pl.LightningModule):
#     r"""
#     Class for distance featurisation
#
#     """
#
#     def __init__(self, step=0.1, lower_bound=0, upper_bound=30, gamma=10):
#         super(RbfExpand, self).__init__()
#
#         self.lower_bound = lower_bound
#         self.upper_bound = upper_bound
#         self.gamma = gamma
#         self.step = step
#         self.spaced_values = torch.arange(
#             self.lower_bound, self.upper_bound, self.step
#         ).to("cuda:0")
#
#     def forward(self, distances):
#         distances = distances.unsqueeze(-1)
#         return torch.exp(-self.gamma * torch.pow((distances - self.spaced_values), 2))
#
#
# class InterBlock(nn.Module):
#     def __init__(self):
#         super(InterBlock, self).__init__()
#         self.conv = ContConv()
#
#         self.atom_wise_list = nn.Sequential(
#             nn.Linear(128, 128), nn.Linear(128, 128), nn.Linear(128, 128)
#         )  # [nn.Linear(64, 64)]
#
#     def forward(
#         self, x: torch.Tensor, r: torch.Tensor, edge_index: torch.Tensor
#     ) -> torch.Tensor:
#         r"""Forward pass through the Interaction block.
#         Parameters
#         ----------
#         x:
#             Embedded properties of charges of the system (n-atoms, n_hidden)
#         r:
#             interatomic distances (n_edges, )
#
#
#         """
#         x = self.atom_wise_list[0](x)
#         # print('x shape ', x.shape)
#         # print(type(x), type(r))
#         x = self.conv(x, r, edge_index)
#         x = self.atom_wise_list[1](x)
#         x = F.tanh(x)  # torch.log(0.5 * torch.exp(x) + 0.5)
#         x = self.atom_wise_list[2](x)
#
#         return x
#
#
# class DistilBertAppl(pl.LightningModule):
#     def __init__(self, batch_size=32, hidden_s=128):
#         super(DistilBertAppl, self).__init__()
#         self.hidden_s = hidden_s
#         self.config = DistilBertConfig(
#             vocab_size=6,
#             max_position_embeddings=29,
#             dim=self.hidden_s,
#             num_labels=1,
#             n_heads=4,
#             **{"problem_type": "regression"}
#         )
#         self.bert = DistilBertForSequenceClassification(self.config)
#         self.emb = nn.Embedding(
#             num_embeddings=6, embedding_dim=self.hidden_s, padding_idx=0
#         )
#         # self.conv = ContConv()
#         self.interaction_list = nn.Sequential(InterBlock(), InterBlock(), InterBlock())
#         self.batch_size = batch_size
#
#     def forward(self, sample):
#
#         x = self.emb(sample.modif_z)
#         # x = self.conv(x, sample.edge_attr.reshape(-1).float(), sample.edge_index)
#         for interaction_block in self.interaction_list:
#             x = interaction_block(
#                 x, sample.edge_attr.reshape(-1).float(), sample.edge_index
#             )
#         res = self.bert(
#             inputs_embeds=x.reshape(self.batch_size, -1, self.hidden_s),
#             attention_mask=sample.attent_mask.reshape(self.batch_size, -1),
#             labels=sample.y[:, 7],
#         )
#         return res["loss"]
#
#     def training_step(self, train_batch, batch_idx):
#         loss = self.forward(train_batch)
#         self.log("train_loss", loss)
#         return loss
#
#     def validation_step(self, val_batch, batch_idx):
#         loss = self.forward(val_batch)
#         self.log("val_loss", loss)
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
#         # sched = torch.optim.lr_scheduler.StepLR(optimizer, 100000,
#         #                                         0.96)  # torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
#         return [optimizer]  # , [sched]
