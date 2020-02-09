from metric.ndcg import ndcg_score_tensor
import torch
import math
from collections import defaultdict

def train_step(model, optimizer, dataset):
    model.train()
    epoch_loss_ls = []
    for torch_batch_rankings, torch_batch_std_labels in dataset:
        loss = model(torch_batch_rankings, torch_batch_std_labels)
        epoch_loss_ls.append(loss.item())
        model.zero_grad()
        loss.backward()
        optimizer.step()
    return sum(epoch_loss_ls) / len(epoch_loss_ls)


def valid_test_step(model, dataset):
    model.eval()
    ndcg_ls = defaultdict(list)
    ndcg_k = {}
    denominator = defaultdict(list)
    for doc_data, y in dataset:
        label, data = y, doc_data
        pred = model.predict(data)
        pred_ar = pred.squeeze(1)
        label_ar = label
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
