import torch.nn as nn
import torch
import torch_geometric
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_scatter import scatter
import pytorch_lightning  as pl
import numpy as np
from transformers import DistilBertConfig, DistilBertForSequenceClassification


class CosineCutoff(nn.Module):
    r"""Class of Behler cosine cutoff.
    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}
    Args:
        cutoff (float, optional): cutoff radius.
    """

    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def forward(self, distances):
        """Compute cutoff.
        Args:
            distances (torch.Tensor): values of interatomic distances.
        Returns:
            torch.Tensor: values of cutoff function.
        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs

class ContConv(MessagePassing):
    r'''
    Filtering network
    '''

    def __init__(self):
        super(ContConv, self).__init__()
        self.rbf = RbfExpand()
        self.dense = [nn.Linear(300, 128), \
                      nn.Linear(128, 128)]
        self.cutoff = CosineCutoff()

    def forward(self, x: torch.Tensor,
                r: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        r"""Forward pass through the Conv block.
                               Parameters
                               ----------
                               x:
                                   Embedded properties of charges of the system (n-atoms, n_hidden)
                               r:
                                   interatomic distances (n_edges, )


                               """
        # print('R do', r.shape)
        C = self.cutoff(r)
        r = self.rbf(r)
        # print('R posle', r.shape)

        for dense_instance in self.dense:
            r = dense_instance(r)
            r = F.tanh(r)#torch.log(0.5 * torch.exp(r) + 0.5)
        r = r * C.unsqueeze(-1)
        prop = self.propagate(edge_index, x=x, W=r, size=None)
        # print(r.shape, x.shape, prop.shape)
        return prop  # x * r.view(-1, 1)

    def message(self, x_j: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return x_j * W


class RbfExpand(nn.Module):
    r'''
    Class for distance featurisation

    '''

    def __init__(self, step=0.1, lower_bound=0, upper_bound=30, gamma=10):
        super(RbfExpand, self).__init__()

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.gamma = gamma
        self.step = step
        self.spaced_values = torch.arange(self.lower_bound, self.upper_bound, self.step).cuda()

    def forward(self, distances):
        distances = distances.unsqueeze(-1)
        return torch.exp(-self.gamma * torch.pow((distances - self.spaced_values), 2))


class DistilBertAppl(pl.LightningModule):

    def __init__(self, batch_size=32, hidden_s=128):
        super(DistilBertAppl, self).__init__()
        self.hidden_s = hidden_s
        self.config = DistilBertConfig(vocab_size=6, max_position_embeddings=29, dim=self.hidden_s, num_labels=1, n_heads=4,
                              **{'problem_type': 'regression'})
        self.bert = DistilBertForSequenceClassification(self.config)
        self.emb = nn.Embedding(num_embeddings=6, embedding_dim=self.hidden_s, padding_idx=0)
        self.conv = ContConv()
        self.batch_size = batch_size

    def forward(self, sample):

        sample.to('cuda:0')
        x = self.emb(sample.modif_z)
        x = self.conv(x, sample.edge_attr.reshape(-1).float(), sample.edge_index)
        res = self.bert(inputs_embeds=x.reshape(self.batch_size, -1, self.hidden_s), \
          attention_mask=sample.attent_mask.reshape(self.batch_size, -1), \
          labels=sample.y[:, 7])
        return res['loss']

    def training_step(self, train_batch, batch_idx):
        loss = self.forward(train_batch)
        self.log('train_loss', loss)
        return loss


    def validation_step(self, val_batch, batch_idx):
        loss = self.forward(val_batch)
        self.log('val_loss', loss)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        # sched = torch.optim.lr_scheduler.StepLR(optimizer, 100000,
        #                                         0.96)  # torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        return [optimizer]#, [sched]