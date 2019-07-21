# -- coding: utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.dataset_trans import torch_batch_triu
from torch.nn.init import xavier_normal_ as nr_init
from config.activation import activation



class RankNet(nn.Module):
    def __init__(self, f_para_dict=None):
        super(RankNet, self).__init__()
        self.model = self.ini_ffnns(**f_para_dict)

    def ini_ffnns(self, input_dim=None, h_dim=None, out_dim=None, num_layers=None, HD_AF=None, HN_AF=None, TL_AF=None, drop_rate=None, apply_tl_af=False):
        head_AF, hidden_AF, tail_AF = activation(HD_AF), activation(HN_AF), activation(TL_AF)

        ffnns = nn.Sequential()
        if 1 == num_layers:
            nr_h1 = nn.Linear(input_dim, out_dim)  # 入力層
            # .weight でパラメータの中身が確認できる
            nr_init(nr_h1.weight)
            ffnns.add_module('L_1', nr_h1)

            if apply_tl_af:
                ffnns.add_module('ACT_1', tail_AF)
        else:
            nr_h1 = nn.Linear(input_dim, h_dim)
            nr_init(nr_h1.weight)
            ffnns.add_module('L_1', nr_h1)
            ffnns.add_module('ACT_1', head_AF)

            if num_layers > 2:           # 隠れ層
                for i in range(2, num_layers):
                    h_dim_half = h_dim / 2
                    ffnns.add_module('_'.join(['DR', str(i)]), nn.Dropout(drop_rate))
                    nr_hi = nn.Linear(h_dim, int(h_dim_half))
                    nr_init(nr_hi.weight)
                    ffnns.add_module('_'.join(['L', str(i)]), nr_hi)
                    ffnns.add_module('_'.join(['ACT', str(i)]), hidden_AF)
                    h_dim = int(h_dim_half)
            nr_hn = nn.Linear(int(h_dim_half), out_dim)  #出力層
            nr_init(nr_hn.weight)
            ffnns.add_module('_'.join(['L', str(num_layers)]), nr_hn)
            if apply_tl_af:
                ffnns.add_module('_'.join(['ACT', str(num_layers)]), tail_AF)

        return ffnns


    def forward(self, torch_batch_rankings, torch_batch_std_labels):

        # モデルによる予測値からペアを作る
        batch_pred = self.model(torch_batch_rankings)  # batch_pred = [40,1]
        batch_pred_dim = torch.squeeze(batch_pred, 1) # batch_pred_dim = [40]
        batch_pred_diffs = batch_pred - torch.unsqueeze(batch_pred_dim, 0)  # batch_pred_diffs = [40, 40]
        batch_s_ij = torch_batch_triu(batch_mats=batch_pred_diffs, k=1,
                                                                  pair_type="All")  # batch_s_ij = [780]

        # ラベルの関連度からペアを作る
        batch_std = torch_batch_std_labels # batch_std = [40]
        batch_std_diffs = torch.unsqueeze(batch_std, 1) - torch.unsqueeze(batch_std, 0)  # batch_std_diffs = [40, 40]
        batch_s_ij_label = torch_batch_triu(batch_mats=batch_std_diffs, k=1,
                                                                  pair_type="All")# batch_s_ij_label = [780]

        # モデルによる予測値からペアを作る
        # batch_pred = self.model(torch_batch_rankings)  # batch_pred = [40,1]
        # batch_pred_reverse = torch.squeeze(batch_pred, 1)
        # batch_pred_reverse = torch.unsqueeze(batch_pred_reverse, 0)  # batch_pred_reverse = [1, 40]
        # batch_pred_diffs = batch_pred - batch_pred_reverse  # batch_pred_diffs = [40, 40]
        # batch_pred_mat = torch.unsqueeze(batch_pred_diffs, 0)  # batch_pred_mat = [1, 40, 40]
        # batch_s_ij = torch_batch_triu(batch_mats=batch_pred_mat, k=1,
        #                               pair_type="All")  # batch_s_ij = [1, 780]
        #
        # # ラベルの関連度からペアを作る
        # batch_std = torch_batch_std_labels
        # batch_std_diffs = torch.unsqueeze(batch_std, 1) - torch.unsqueeze(batch_std, 0)  # batch_std_diffs = [40, 40]
        # batch_std_mat = torch.unsqueeze(batch_std_diffs, 0)  # batch_std_mat = [1, 40, 40]
        # batch_s_ij_label = torch_batch_triu(batch_mats=batch_std_mat, k=1,
        #                                     pair_type="All")  # batch_s_ij_label = [1, 780]

        #print(min(batch_s_ij_label))
        # ラベルペアのなかで1より大きい場合は1に, -1未満は-1に揃える
        batch_Sij = torch.clamp(batch_s_ij_label, -1, 1)

        sigma = 1.0
        batch_loss_1st = 0.5 * sigma * batch_s_ij * (1.0 - batch_Sij)
        batch_loss_2nd = torch.log(torch.exp(-sigma * batch_s_ij) + 1.0)
        batch_loss = torch.sum(batch_loss_1st + batch_loss_2nd) / batch_s_ij.size(0)


        return batch_loss

    def predict(self, x):
        return self.model(x)


