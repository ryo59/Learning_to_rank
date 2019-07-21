# -- coding: utf-8 --

import torch
import os
from dataset.dataset_trans import L2RDataset
from model.train import train_step, valid_test_step
from model.RankNet.ranknet import RankNet
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    k_fold = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
    sum_fold_ndcg_k = defaultdict(dict)

    for fold in k_fold:
        # af = ['ReLU', 'LReLU', 'ReLU6', 'Sig', 'Tan']
        model = RankNet(f_para_dict={'input_dim': 46, 'h_dim': 128, 'out_dim': 1, \
                                     'num_layers': 4, 'HD_AF': "ReLU", 'HN_AF': "ReLU", 'TL_AF': "Sig",
                                     'drop_rate': 0.01})
        model_write = str(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        epochs =100
        # batch_size = 64
        logs_dir = './logs_model'
        best_ndcg = {}
        model_file = {}
        loss_list = []
        val_dcg_list = []
        writer = SummaryWriter(log_dir="logs_0712_dr0.5")

        train_file = os.path.join("/Users/inagaki_ryo/dataset/MQ2007", fold, "train.txt")
        vali_file = os.path.join("/Users/inagaki_ryo/dataset/MQ2007", fold, "vali.txt")
        test_file = os.path.join("/Users/inagaki_ryo/dataset/MQ2007", fold, "test.txt")

        train_dataset = L2RDataset(file=train_file, data_id='MQ2007_Super')
        vali_dataset = L2RDataset(file=vali_file, data_id='MQ2007_Super')
        test_dataset = L2RDataset(file=test_file, data_id='MQ2007_Super')

        for epoch in range(epochs):

            epoch_train_loss = train_step(model, optimizer, train_dataset)
            epoch_train_ndcg = valid_test_step(model, train_dataset)
            epoch_valid_ndcg = valid_test_step(model, vali_dataset)
            for k in [1,3,5,10]:
                print("{} Epoch: {} Train Loss: {} Train NDCG@{}: {} Valid NDCG@{} score: {}".format(fold, epoch, epoch_train_loss,
                                                                                k, epoch_train_ndcg[k], k, epoch_valid_ndcg[k]))

                writer.add_scalar("{} Train loss".format(fold), epoch_train_loss, epoch)
                writer.add_scalar("{} Train NDCG@{} score".format(fold, k), epoch_train_ndcg[k], epoch)
                writer.add_scalar("{} Valid NDCG@{} score".format(fold, k), epoch_valid_ndcg[k], epoch)

                best_ndcg.setdefault(k, 0.0)

                if epoch_valid_ndcg[k] > best_ndcg[k]:
                    #print('--------%s valid_NDCG improved from %.5f to %.5f!--------' % (fold, best_ndcg[k], epoch_valid_ndcg[k]))
                    best_ndcg[k] = epoch_valid_ndcg[k]
                    model_file[k] = '%s-epoch%03d-NDCG@%s：%.5f.pth' % (fold, epoch, k, best_ndcg[k])
                    torch.save(model.state_dict(), os.path.join(logs_dir, model_file[k]))

            print("--" * 50)

        for k in [1, 3, 5, 10]:
            print("**We use {} for test**".format(model_file[k]))
            weight_file = os.path.join(logs_dir, model_file[k])
            model.load_state_dict(torch.load(weight_file))
            epoch_test_ndcg = valid_test_step(model, test_dataset)
            print("{} Test NDCG@{}: {}".format(fold, k, epoch_test_ndcg[k]))
            print("--" * 50)


            sum_fold_ndcg_k[k][fold] = float(epoch_test_ndcg[k])

    with open("result_dr05.txt", "w") as f:
        f.write("Model Architecture：\n{}\nEpoch：{}\n\n".format(model_write, epochs))

        for k in [1, 3, 5, 10]:
            ave_kfold_ndcg = {}
            #print(sum_fold_ndcg_k)
            ave_kfold_ndcg = sum(sum_fold_ndcg_k[k].values()) / float(len(k_fold))
            for fold in k_fold:
                print("{} Test NDCG@{}：{}".format(fold, k, sum_fold_ndcg_k[k][fold]))
                f.write("{} Test NDCG@{}：{} \n".format(fold, k, sum_fold_ndcg_k[k][fold]))
            print("Average Test NDCG@{}: {}".format(k, ave_kfold_ndcg))
            f.write("Average Test NDCG@{}: {}\n\n".format(k, ave_kfold_ndcg))
        f.close()
    writer.close()
