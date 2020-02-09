import torch
import numpy as np


# For LambdaRank
def dcg(sorted_labels):
    gains = torch.pow(2.0, sorted_labels) - 1.0
    discounts = torch.log2(torch.arange(sorted_labels.size(0)).type(torch.FloatTensor) + 2.0)
    dcgs = torch.sum(gains / discounts)

    return dcgs

# NDCG for tensor(input:tensor)
def ndcg_score_tensor(y_true, y_pred, k, gains="exponential"):

    def dcg_score_tensor(sorted_label, k, gains="exponential"):
        sorted_label_k = sorted_label[:k].float()

        if gains == "exponential":
            gains = torch.pow(2.0, sorted_label_k) - 1.0

        else:
            raise ValueError("Invalid gains option.")

        discounts = torch.log2(torch.arange(k).type(torch.FloatTensor) + 2.0)

        return torch.sum(gains / discounts)

    best = dcg_score_tensor(y_true, k, gains)
    actual = dcg_score_tensor(y_pred, k, gains)

    return (actual / best).item()

if __name__ == '__main__':
   sys_sorted_labels = [1, 1, 0, 1, 0, 1, 0, 0]
   ideal_sorted_labels=[1, 1, 1, 1, 0, 0, 0, 0]
   sys_sorted_labels = torch.from_numpy(np.asarray(sys_sorted_labels))
   ideal_sorted_labels = torch.from_numpy(np.asarray(ideal_sorted_labels))
   for k in [1, 3, 4, 6, 8, 10]:
       print("NDCG@{}".format(k), ndcg_score_tensor(ideal_sorted_labels, sys_sorted_labels, k=k))


