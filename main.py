import torch
import os
from utils.dataset_trans import L2RDataset
from model.train import train_step, valid_test_step
from model.get_model import get_ranking_model
from config.arg import ArgsUtil
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from utils.make_graph import make_loss_ndcg_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args_obj = ArgsUtil()
    l2r_args = args_obj.get_l2r_args()
    k_fold = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5"]
    torch.manual_seed(1)
    sum_fold_ndcg_k = defaultdict(dict)
    start_time = time.time()
    tensorboard_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    writer = SummaryWriter(log_dir="./logs/{}/{}/tensorboard_logs_{}".format(l2r_args.data, l2r_args.model, tensorboard_date))
    if not os.path.exists("./logs/logs_model"): os.makedirs("./logs/logs_model")
    logs_dir = "./logs/logs_model"

    for fold in k_fold:

        model = get_ranking_model(l2r_args.model, para={'input_dim': l2r_args.input_dim, 'h_dim': l2r_args.h_input_size, 'num_layers': l2r_args.num_layers, \
                                            'hd_af': l2r_args.hd_af, 'hn_af': l2r_args.hn_af, 'tl_af': l2r_args.tl_af, \
                                            'apply_tl_af': l2r_args.apply_tl_af, 'dropout_rate': l2r_args.dropout_rate})

        #model = model.to(device)
        model_write = str(model)
        optimizer = l2r_args.optim(model.parameters(), lr=l2r_args.learning_rate)
        epochs =l2r_args.num_epochs
        best_ndcg = defaultdict(int)
        model_file = {}

        train_file = os.path.join(l2r_args.dir_data, fold, "train.txt")
        vali_file = os.path.join(l2r_args.dir_data, fold, "vali.txt")
        test_file = os.path.join(l2r_args.dir_data, fold, "test.txt")

        train_dataset = L2RDataset(file=train_file, data_id=l2r_args.data)
        print("Finish loadding Train dataset")
        vali_dataset = L2RDataset(file=vali_file, data_id=l2r_args.data)
        print("Finish loadding Valid dataset")
        test_dataset = L2RDataset(file=test_file, data_id=l2r_args.data)
        print("Finish loadding Test dataset")

        epoch_train_loss_list = []
        epoch_valid_ndcg_list = defaultdict(list)

        for epoch in range(epochs):

            epoch_train_loss = train_step(model, optimizer, train_dataset)
            epoch_train_ndcg = valid_test_step(model, train_dataset)
            epoch_valid_ndcg = valid_test_step(model, vali_dataset)

            epoch_train_loss_list.append(epoch_train_loss)

            for k in [1,3,5,10]:

                epoch_valid_ndcg_list[k].append(epoch_valid_ndcg[k])

                print("{} Epoch: {} Train Loss: {} Train NDCG@{}: {} Valid NDCG@{} score: {}".format(fold, epoch, epoch_train_loss,
                                                                                k, epoch_train_ndcg[k], k, epoch_valid_ndcg[k]))

                writer.add_scalar("{} Train loss".format(fold), epoch_train_loss, epoch)
                writer.add_scalar("{} Train NDCG@{} score".format(fold, k), epoch_train_ndcg[k], epoch)
                writer.add_scalar("{} Valid NDCG@{} score".format(fold, k), epoch_valid_ndcg[k], epoch)



                if epoch_valid_ndcg[k] > best_ndcg[k]:
                    best_ndcg[k] = epoch_valid_ndcg[k]
                    date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
                    model_file[k] = "{}-{}-epoch{}-NDCG@{}：{}.pth".format(date, fold, epoch, k, best_ndcg[k])
                    torch.save(model.state_dict(), os.path.join(logs_dir, model_file[k]))

            print("--" * 50)


        #Drawing the state of training
        # for ndcg_index in [1, 3, 5, 10]:
        #     make_loss_ndcg_graph(fold, l2r_args.data, l2r_args.model, epochs, ndcg_index, epoch_train_loss_list, epoch_valid_ndcg_list)


        for k in [1, 3, 5, 10]:
            print("**We use {} for test**".format(model_file[k]))
            weight_file = os.path.join(logs_dir, model_file[k])
            model.load_state_dict(torch.load(weight_file))
            epoch_test_ndcg = valid_test_step(model, test_dataset)
            print("{} Test NDCG@{}: {}".format(fold, k, epoch_test_ndcg[k]))
            print("--" * 50)


            sum_fold_ndcg_k[k][fold] = float(epoch_test_ndcg[k])

        finish_time = time.time()
        elapse = finish_time - start_time

    date_result = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    if not os.path.exists("./result/{}/{}".format(l2r_args.data, l2r_args.model)): os.makedirs("./result/{}/{}".format(l2r_args.data, l2r_args.model))
    with open("./result/{}/{}/result_{}.txt".format(l2r_args.data, l2r_args.model, date_result), "w") as f:
        f.write("Model Architecture：\n{}\nEpoch：{}\nElapsed time：{}\n\n".format(model_write, epochs, elapse))

        for k in [1, 3, 5, 10]:
            ave_kfold_ndcg = {}
            ave_kfold_ndcg = sum(sum_fold_ndcg_k[k].values()) / float(len(k_fold))
            for fold in k_fold:
                print("{} Test NDCG@{}：{}".format(fold, k, sum_fold_ndcg_k[k][fold]))
                f.write("{} Test NDCG@{}：{} \n".format(fold, k, sum_fold_ndcg_k[k][fold]))
            print("Average Test NDCG@{}: {}".format(k, ave_kfold_ndcg))
            f.write("Average Test NDCG@{}: {}\n\n".format(k, ave_kfold_ndcg))
        f.close()
    writer.close()
