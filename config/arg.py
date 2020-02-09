
import argparse
import torch

class ArgsUtil(object):
    def __init__(self, given_root=None):
        self.args_parser = argparse.ArgumentParser('Run learning to rank for pytorch')
        self.ini_l2r_args()

    def ini_l2r_args(self):


        '''data'''
        self.args_parser.add_argument('--data', help='the data collection upon which you will perform learning-to-rank.')
        self.args_parser.add_argument('--dir_data', help='the path where the data locates.')

        ''' train '''
        self.args_parser.add_argument('--num_epochs', type=int, default=100, help='the number of training epoches.(default: 100)')
        self.args_parser.add_argument('--learning_rate', type=float, default=0.001, help='the magnitude of the learning rate.(default: 0.001)')

        ''' model-specific '''
        self.args_parser.add_argument('--model', help='select learning to rank model', choices=['RankNet', 'LambdaRank'])
        self.args_parser.add_argument('--optim', default=torch.optim.Adam, help='select Optimizer(default: Adam)')
        self.args_parser.add_argument('--input_dim', type=int, default=46,
                                      help='the input size of the first layer of the hidden layer.(default: 46)')
        self.args_parser.add_argument('--h_input_size', type=int, default=128,
                                      help='the input size of the first layer of the hidden layer.(default: 128)')
        self.args_parser.add_argument('--num_layers', type=int, default=4, help='the number of layers.(default: 4)')
        self.args_parser.add_argument('--hd_af', default="ReLU",
                                      help='the type of activation function for head layers.(default: ReLU)["ReLU", "LReLU", "ReLU6", "Sig", "Tan"]')
        self.args_parser.add_argument('--hn_af', default="ReLU",
                                      help='the type of activation function for hidden layers.(default: ReLU)')
        self.args_parser.add_argument('--tl_af', default="Sig",
                                      help='the type of activation function for the final layer.(default: Sigmoid)')
        self.args_parser.add_argument('--apply_tl_af', action='store_true',
                                      help='perform activation for the final layer.')
        self.args_parser.add_argument('--dropout_rate', type=float, default=0.01,
                                      help='the magnitude of the dropout rate.(default: 0.01)')


    def get_l2r_args(self):
        l2r_args = self.args_parser.parse_args()
        return l2r_args
