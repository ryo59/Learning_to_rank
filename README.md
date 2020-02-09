# Learning_to_rank
Learning to rank in Pytorch
### Dataset
[MQ2007, 2008](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/)  
[MSLR-WEB10K, 30K](https://www.microsoft.com/en-us/research/project/mslr/)

### Prerequisites
- Python 3.6
- PyTorch 1.1.0
- tb-nightly, future # for tensorboard

### Usage
1. Download the dataset ([MQ2007, 2008](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/) or [MSLR-WEB10K, 30K](https://www.microsoft.com/en-us/research/project/mslr/))
2. Train RankNet or LambdaRank:

```
python main.py --data MQ2007 --dir_data "/$DATA_PATH/MQ2007" --model RankNet
```  

```
python main.py -h

optional arguments:
  -h, --help                       show this help message and exit
  --data DATA                      the data collection upon which you will perform learning-to-rank.
  --dir_data DIR_DATA              the path where the data locates.
  --num_epochs NUM_EPOCHS          the number of training epoches.(default: 100)
  --learning_rate LEARNING_RATE    the magnitude of the learning rate.(default: 0.001)
  --model {RankNet,LambdaRank}     select learning to rank model
  --optim OPTIM                    select Optimizer(default: Adam)
  --input_dim INPUT_DIM            the input size of the first layer of the hidden layer.(default: 46)
  --h_input_size H_INPUT_SIZE      the input size of the first layer of the hidden layer.(default: 128)
  --num_layers NUM_LAYERS          the number of layers.(default: 4)
  --hd_af HD_AF                    the type of activation function for head layers.(default: ReLU)["ReLU", "LReLU", "ReLU6",
                                   "Sig", "Tan"]
  --hn_af HN_AF                    the type of activation function for hidden layers.(default: ReLU)
  --tl_af TL_AF                    the type of activation function for the final layer.(default: Sigmoid)
  --apply_tl_af                    perform activation for the final layer.
  --dropout_rate DROPOUT_RATE      the magnitude of the dropout rate.(default: 0.01)
```

### Reference
[From RankNet to LambdaRank to LambdaMART: An Overview](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.180.634&rep=rep1&type=pdf)
