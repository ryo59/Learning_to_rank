import matplotlib.pyplot as plt
import os

def make_loss_ndcg_graph(fold, dataset, model, epoch, index, epoch_train_loss_list, epoch_valid_ndcg_list):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(range(epoch), epoch_train_loss_list, 'r-', label='train_loss')
    ax2 = ax1.twinx()  # 2つのプロットを関連付ける
    ax2.plot(range(epoch), epoch_valid_ndcg_list[index], 'b-', label='NDCG@{}'.format(index))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='lower left')

    ax1.set_title("{}_{}_Trainloss&NDCG@{}".format(model, fold, index))
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('train_loss')
    ax2.set_ylabel('NDCG@{}'.format(index))
    ax1.grid(True)

    plt.show()
    if not os.path.exists("images/{}/{}/NDCG@{}".format(dataset, model, index)): os.makedirs("images/{}/{}/NDCG@{}".format(dataset, model, index))
    plt.savefig("images/{}/{}/NDCG@{}/{}_loss_ndcg.png".format(dataset, model, index, fold), bbox_inches="tight")
