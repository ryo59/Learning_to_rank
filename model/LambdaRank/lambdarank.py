# -- coding: utf-8 --
import torch
import torch.nn as nn
from eval.metrics import dcg
from torch.nn.init import xavier_normal_ as nr_init
from config.activation import activation



class LambdaRank(nn.Module):
    def __init__(self, f_para_dict=None):
        super(LambdaRank, self).__init__()
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
        batch_pred_dim = torch.squeeze(batch_pred, 1)  # batch_pred_dim = [40]
        batch_pred_diffs = batch_pred - torch.unsqueeze(batch_pred_dim, 0)  # batch_pred_diffs = [40, 40]

        # ラベルの関連度からペアを作る
        batch_std = torch_batch_std_labels  # batch_std = [40]
        batch_std_diffs = torch.unsqueeze(batch_std, 1) - torch.unsqueeze(batch_std, 0)  # batch_std_diffs = [40, 40]

        # ラベルペアのなかで1より大きい場合は1に, -1未満は-1に揃える
        batch_Sij = torch.clamp(batch_std_diffs, -1, 1)

        sigma = 1.0
        batch_loss_1st = 0.5 * sigma * batch_pred_diffs * (1.0 - batch_Sij)
        batch_loss_2nd = torch.log(torch.exp(-sigma * batch_pred_diffs) + 1.0)


        # ΔNDCGの計算
        batch_preds_sorted, batch_preds_sorted_inds = torch.sort(batch_pred_dim, descending=True)
        batch_stds_sorted_by_preds = torch_batch_std_labels[batch_preds_sorted_inds]

        #G
        batch_dcgs = dcg(torch_batch_std_labels)
        batch_gains = torch.pow(2.0, batch_stds_sorted_by_preds) - 1.0
        batch_gains_normalize = batch_gains / batch_dcgs #batch_n_gains = [40]

        #D
        batch_discounts = torch.log2(torch.arange(batch_stds_sorted_by_preds.size(0)).type(torch.FloatTensor) + 2.0)

        delta_ndcg_abs = torch.abs(batch_gains_normalize) / torch.abs(batch_discounts) #delta_ndcg_abs = [40]
        delta_ndcg_abs_diff = torch.unsqueeze(delta_ndcg_abs, 1) - torch.unsqueeze(delta_ndcg_abs, 0)

        # 対角線以外でlossを計算
        diagona = 1 - torch.eye(batch_loss_1st.shape[0])
        batch_loss = ((batch_loss_1st + batch_loss_2nd) * delta_ndcg_abs_diff) * diagona
        combination = (batch_loss_1st.shape[0] * (batch_loss_1st.shape[0] - 1)) / 2
        batch_loss_triu = (torch.sum(batch_loss) / 2) / combination

        return batch_loss_triu

    def predict(self, x):
        return self.model(x)
