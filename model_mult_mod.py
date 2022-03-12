import torch.nn as nn
import torch
import torch_geometric
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_scatter import scatter
import pytorch_lightning  as pl
import numpy as np
from transformers import BertModel, BertConfig, MMBTConfig, MMBTForClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import CrossEntropyLoss, MSELoss


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

class Helper(MMBTForClassification):

  def forward(
          self,
          input_modal,
          input_ids=None,
          modal_start_tokens=None,
          modal_end_tokens=None,
          attention_mask=None,
          token_type_ids=None,
          modal_token_type_ids=None,
          position_ids=None,
          modal_position_ids=None,
          head_mask=None,
          inputs_embeds=None,
          labels=None,
          return_dict=None,
          encoder_attention_mask=None
      ):
          return_dict = return_dict if return_dict is not None else self.config.use_return_dict

          outputs = self.mmbt(
              input_modal=input_modal,
              input_ids=input_ids,
              modal_start_tokens=modal_start_tokens,
              modal_end_tokens=modal_end_tokens,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids,
              modal_token_type_ids=modal_token_type_ids,
              position_ids=position_ids,
              modal_position_ids=modal_position_ids,
              head_mask=head_mask,
              inputs_embeds=inputs_embeds,
              return_dict=return_dict,
              encoder_attention_mask=encoder_attention_mask
          )

          pooled_output = outputs[1]

          pooled_output = self.dropout(pooled_output)
          logits = self.classifier(pooled_output)

          loss = None
          if labels is not None:
              if self.num_labels == 1:
                  #  We are doing regression
                  loss_fct = MSELoss()
                  loss = loss_fct(logits.view(-1), labels.view(-1))
              else:
                  loss_fct = CrossEntropyLoss()
                  loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

          if not return_dict:
              output = (logits,) + outputs[2:]
              return ((loss,) + output) if loss is not None else output

          return SequenceClassifierOutput(
              loss=loss,
              logits=logits,
              hidden_states=outputs.hidden_states,
              attentions=outputs.attentions,
          )



class MultiMod(pl.LightningModule):

    def __init__(self, batch_size=32, hidden_s=64):
        super(MultiMod, self).__init__()
        self.hidden_s = hidden_s
        self.config = BertConfig(vocab_size=6, max_position_embeddings=100, hidden_size=64, num_labels=1, num_attention_heads=4,
                       num_hidden_layers=6,
                       **{'problem_type': 'regression'})
        self.bert = BertModel(self.config)
        self.emb = nn.Embedding(num_embeddings=6, embedding_dim=self.hidden_s, padding_idx=0)
        self.rbf = RbfExpand()

        self.config_MM = MMBTConfig(self.config, num_labels=1, modal_hidden_size=300)
        self.mm_transformer = Helper(self.config_MM, encoder=self.rbf, transformer=self.bert)

        self.batch_size = batch_size

    def forward(self, sample):

        x = self.emb(sample.modif_z).reshape(self.batch_size, -1, self.hidden_s)
        #x = self.conv(x, sample.edge_attr.reshape(-1).float(), sample.edge_index)
        res = self.mm_transformer(input_modal=sample.distances_padded.reshape(self.batch_size, -1),
                                  inputs_embeds=x,  #ample.reshape(self.batch_size, -1, self.hidden_s), \
          attention_mask=sample.attent_mask.reshape(self.batch_size, -1), \
                         encoder_attention_mask = sample.attent_dist.reshape(self.batch_size, -1),
          labels=sample.y[:, 7], return_dict=True)
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