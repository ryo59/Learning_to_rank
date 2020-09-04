# -- coding: utf-8 --
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from utils.dataset_trans import L2RDataset
from model.train import train_step, valid_test_step
from model.get_model import get_ranking_model
from config.arg import ArgsUtil
from config.activation import activation
from torch.nn.init import xavier_normal_ as nr_init
from metric.ndcg import ndcg_score_tensor
import math
from collections import defaultdict

optuna.logging.disable_default_handler()

args_obj = ArgsUtil()
l2r_args = args_obj.get_l2r_args()

train_file = os.path.join(l2r_args.dir_data, "Fold1", "train.txt")
test_file = os.path.join(l2r_args.dir_data, "Fold1", "test.txt")

train_loader =  L2RDataset(file=train_file, data_id=l2r_args.data)
test_loader = L2RDataset(file=test_file, data_id=l2r_args.data)

#モデルの定義
class RankNet(nn.Module):
    def __init__(self, trial, num_layer, input_dim, h_dim, lr_rate):
        super(RankNet, self).__init__()
        self.activation = get_activation(trial)
        # 第1層
        self.fc = nn.ModuleList([nn.Linear(input_dim, h_dim[0])])
        # 第2層以降
        for i in range(1, num_layer):
            self.fc.append(nn.Linear(h_dim[i-1], h_dim[i]))
        self.dropout = nn.Dropout(lr_rate)

        self.fc_last = nn.Linear(h_dim[i], 1)

    def forward(self, x):
        for i, l in enumerate(self.fc):
            x = self.dropout(x)
            nr_init(x.weight)
            x = l(x)
            x = self.activation(x)
        x = self.fc_last(x)
        return x

    def loss(self, torch_batch_rankings, torch_batch_std_labels):

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


def train(model, device, train_loader, optimizer):
  model.train()
  for torch_batch_rankings, torch_batch_std_labels in train_loader:
        data, target = torch_batch_rankings.to(device), torch_batch_std_labels.to(device)
        optimizer.zero_grad()
        loss = model.loss(data, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    ndcg_ls = defaultdict(list)
    ndcg_k = {}
    denominator = defaultdict(list)
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        pred = model.predict(data)
        pred_ar = pred.squeeze(1)
        label_ar = target
        _, order = torch.sort(pred_ar, descending=True)
        y_pred_sorted = label_ar[order]
        for k in [1, 3, 5, 10]:
            if len(label_ar) > k:
                ndcg_s = ndcg_score_tensor(label_ar, y_pred_sorted, k=k)
                if not math.isnan(ndcg_s):
                    ndcg_ls[k].append(ndcg_s)

            else:
                denominator[k].append(len(label_ar))

    for k in [1, 3, 5, 10]:
        # Subtraction for the number of documents less than k
        ndcg_k[k] = sum(ndcg_ls[k]) / (len(ndcg_ls[k]) - len(denominator[k]))

    return ndcg_k

def get_optimizer(trial, model):
  optimizer_names = ['Adam', 'MomentumSGD', 'rmsprop']
  optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
  weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
  if optimizer_name == optimizer_names[0]:
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
    optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
  elif optimizer_name == optimizer_names[1]:
    momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
    optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)
  else:
    optimizer = optim.RMSprop(model.parameters())
  return optimizer

def get_activation(trial):
  activation_names = ['ReLU', 'ELU']
  activation_name = trial.suggest_categorical('activation', activation_names)
  if activation_name == activation_names[0]:
    activation = F.relu
  else:
    activation = F.elu
  return activation

EPOCH = 10
#num_layer, input_dim, h_dim, lr_rate)

def objective(trial):
  device = "cuda" if torch.cuda.is_available() else "cpu"

  #畳み込み層の数
  num_layer = trial.suggest_int('num_layer', 3, 7)

  #各畳込み層のフィルタ数
  h_dim = [int(trial.suggest_discrete_uniform("h_dim_"+str(i), 128, 16, 16)) for i in range(num_layer)]

  lr_rate = [trial.suggest_uniform("dropout_l{}".format(i), 0.2, 0.5)for i in range(num_layer)]

  model = RankNet(trial, num_layer, 46, h_dim, lr_rate).to(device)
  optimizer = get_optimizer(trial, model)

  for step in range(EPOCH):
    train(model, device, train_loader, optimizer)
    ndcg = test(model, device, test_loader)

  return ndcg

if __name__ == '__main__':
  TRIAL_SIZE = 100
  study = optuna.create_study()
  study.optimize(objective, n_trials=TRIAL_SIZE)
