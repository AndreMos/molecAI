import torch.nn as nn
import torch
import torch_geometric
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_scatter import scatter
import pytorch_lightning  as pl

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
        self.dense = [nn.Linear(300, 64), \
                      nn.Linear(64, 64)]
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
        self.spaced_values = torch.arange(self.lower_bound, self.upper_bound, self.step)

    def forward(self, distances):
        distances = distances.unsqueeze(-1)
        return torch.exp(-self.gamma * torch.pow((distances - self.spaced_values), 2))


class InterBlock(nn.Module):

    def __init__(self):
        super(InterBlock, self).__init__()
        self.conv = ContConv()

        self.atom_wise_list = [nn.Linear(64, 64)] * 3  # [nn.Linear(64, 64)]

    def forward(self, x: torch.Tensor,
                r: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        r"""Forward pass through the Interaction block.
                       Parameters
                       ----------
                       x:
                           Embedded properties of charges of the system (n-atoms, n_hidden)
                       r:
                           interatomic distances (n_edges, )


                       """
        x = self.atom_wise_list[0](x)
        # print('x shape ', x.shape)
        # print(type(x), type(r))
        x = self.conv(x, r, edge_index)
        x = self.atom_wise_list[1](x)
        x = F.tanh(x)#torch.log(0.5 * torch.exp(x) + 0.5)
        x = self.atom_wise_list[2](x)

        return x


class Schnet(pl.LightningModule):
    def __init__(self):
        r"""
        Default structure of Schnet architecture


        """
        super(Schnet, self).__init__()
        self.transform = torch_geometric.transforms.Distance(norm=False)
        self.emb = nn.Embedding(num_embeddings=6, embedding_dim=64)
        self.interaction_list = [InterBlock()] * 3
        self.atom_wise32 = nn.Linear(64, 32)
        self.atom_wise1 = nn.Linear(32, 1)
        self.corr_dict = {1 : 1, 6 : 2, 7 : 3, 8 : 4, 9 : 5}

    def forward(self, sample):
        # graph : torch_geometric.data.Data,
        # charges : torch.Tensor,
        # keys : torch.Tensor): #-> (torch.Tensor, torch.Tensor):
        r"""Forward pass through the SchNet architecture.
                Parameters
                ----------
                graph:
                    Input torch-geom data graph object containing batch atom positions
                charges:
                    Input tensor containing charges of atoms (n_atoms, )

                Returns
                -------
                force:
                   force acting on each atom in the system (n_atoms, 3)
                energy:
                    PMF of the system, scalar
                """
        charges = torch.tensor([self.corr_dict[x.item()] for x in sample.z], dtype=torch.long)
        x = self.emb(charges)
        # self.transform(graph)
        r = sample.edge_attr.reshape(-1).float()
        # print(r.shape, x.shape)
        for interaction_block in self.interaction_list:
            x = interaction_block(x, r, sample.edge_index)
        # print(x.shape)
        x = self.atom_wise32(x)
        x = F.tanh(x)#torch.log(0.5 * torch.exp(x) + 0.5)
        # print(x.shape)
        energy = self.atom_wise1(x)
        # force = -torch.autograd.grad(
        #     energy.sum(),
        #     graph.pos,
        #     retain_graph=True,

        # )[0]

        energy = scatter(energy, sample.batch, dim=0, reduce="sum").flatten()

        return energy

    def mse(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch, train_batch.y[:, 7]
        logits = self.forward(x)
        loss = self.mse(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch, val_batch.y[:, 7]
        logits = self.forward(x)
        loss = self.mse(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        sched = StepLR(optimizer, 100000, 0.96)#torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        return [optimizer], [sched]
