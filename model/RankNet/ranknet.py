import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_ as nr_init
from config.activation import activation



class RankNet(nn.Module):
    def __init__(self, f_para_dict=None):
        super(RankNet, self).__init__()
        self.model = self.ini_ffnns(**f_para_dict)

    def ini_ffnns(self, input_dim=None, h_dim=None, out_dim=1, num_layers=None, hd_af=None, hn_af=None, tl_af=None, dropout_rate=None, apply_tl_af=None):
        head_AF, hidden_AF, tail_AF = activation(hd_af), activation(hn_af), activation(tl_af)

        ffnns = nn.Sequential()
        if 1 == num_layers:
            nr_h1 = nn.Linear(input_dim, out_dim)  # Input layer
            nr_init(nr_h1.weight)
            ffnns.add_module('L_1', nr_h1)

            if apply_tl_af:
                ffnns.add_module('ACT_1', tail_AF)
        else:
            nr_h1 = nn.Linear(input_dim, h_dim)
            nr_init(nr_h1.weight)
            ffnns.add_module('L_1', nr_h1)
            ffnns.add_module('ACT_1', head_AF)

            if num_layers > 2:           # Hidden layer
                for i in range(2, num_layers):
                    h_dim_half = h_dim / 2
                    ffnns.add_module('_'.join(['DR', str(i)]), nn.Dropout(dropout_rate))
                    nr_hi = nn.Linear(h_dim, int(h_dim_half))
                    nr_init(nr_hi.weight)
                    ffnns.add_module('_'.join(['L', str(i)]), nr_hi)
                    ffnns.add_module('_'.join(['ACT', str(i)]), hidden_AF)
                    h_dim = int(h_dim_half)
            nr_hn = nn.Linear(int(h_dim_half), out_dim)  #Output layer
            nr_init(nr_hn.weight)
            ffnns.add_module('_'.join(['L', str(num_layers)]), nr_hn)
            if apply_tl_af:
                ffnns.add_module('_'.join(['ACT', str(num_layers)]), tail_AF)

        return ffnns


    def forward(self, torch_batch_rankings, torch_batch_std_labels):

        # Make a pair from the model predictions
        batch_pred = self.model(torch_batch_rankings)  # batch_pred = [40,1]
        batch_pred_dim = torch.squeeze(batch_pred, 1) # batch_pred_dim = [40]
        batch_pred_diffs = batch_pred - torch.unsqueeze(batch_pred_dim, 0)  # batch_pred_diffs = [40, 40]

        # Make a pair from the relevance of the label
        batch_std = torch_batch_std_labels # batch_std = [40]
        batch_std_diffs = torch.unsqueeze(batch_std, 1) - torch.unsqueeze(batch_std, 0)  # batch_std_diffs = [40, 40]

        # Align to -1 ~ 1
        batch_Sij = torch.clamp(batch_std_diffs, -1, 1)

        sigma = 1.0
        batch_loss_1st = 0.5 * sigma * batch_pred_diffs * (1.0 - batch_Sij)
        batch_loss_2nd = torch.log(torch.exp(-sigma * batch_pred_diffs) + 1.0)

        # Calculate loss outside diagonal
        diagona = 1 - torch.eye(batch_loss_1st.shape[0])
        batch_loss = (batch_loss_1st + batch_loss_2nd) * diagona
        combination = (batch_loss_1st.shape[0] * (batch_loss_1st.shape[0] - 1)) / 2

        batch_loss_triu = (torch.sum(batch_loss) / 2) / combination

        #print(batch_loss_triu)

        return batch_loss_triu

    def predict(self, x):
        return self.model(x)

