# -- coding: utf-8 --

import torch
import numpy as np


# NDCG for tensor(input:tensor)

def ndcg_score_tensor(y_true, y_pred, k, gains="exponential"):

    def dcg_score_tensor(sorted_label, k, gains="exponential"):

        if len(sorted_label) < k:
            sorted_label_k = sorted_label[:len(sorted_label)].float()

            if gains == "exponential":
                gains = torch.pow(2, sorted_label_k) - 1
            else:
                raise ValueError("Invalid gains option.")

            discounts = torch.log2(torch.arange(len(sorted_label), dtype=torch.float) + 2.0)

        else:
            sorted_label_k = sorted_label[:k].float()

            if gains == "exponential":
                gains = torch.pow(2, sorted_label_k) - 1
            else:
                raise ValueError("Invalid gains option.")

            discounts = torch.log2(torch.arange(k, dtype=torch.float) + 2.0)

        return torch.sum(gains / discounts).item()

    best = dcg_score_tensor(y_true, k, gains)
    actual = dcg_score_tensor(y_pred, k, gains)
    if best == 0:
        return 0
    else:
        return (actual / best)



if __name__ == '__main__':
   sys_sorted_labels = [1, 1, 0, 1, 0, 1, 0, 0]
   ideal_sorted_labels=[1, 1, 1, 1, 0, 0, 0, 0]
   sys_sorted_labels = torch.from_numpy(np.asarray(sys_sorted_labels))
   ideal_sorted_labels = torch.from_numpy(np.asarray(ideal_sorted_labels))

   for k in [1, 3, 4, 8]:
    print("NDCG@{}".format(k), ndcg_score_tensor(ideal_sorted_labels, sys_sorted_labels, k=k))


