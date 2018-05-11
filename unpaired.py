import numpy as np
import pip
import torch
from torch import nn
import fastText
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dis_hidden_dim', type=int, default=2048, help='dimension of hidden state')
parser.add_argument('--dis_dropout',type=float, default=0.5, help='dropout value')
parser.add_argument('--dis_input_dropout', type=float, default=0.3, help='Point Number [default: 1024]')

params=parser.parse_args()
class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = 300
        self.dis_layers = 3
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)

def Orthogonal(W,beta):
    W=(1+beta)*W- beta*(W.T*W)*W
    return W

def load_fasttext_model(path):
    """
    Load a binarized fastText model.
    """
    try:
        import fastText
    except ImportError:
        raise Exception("Unable to import fastText. Please install fastText for Python: "
                        "https://github.com/facebookresearch/fastText")
    return fastText.load_model(path)

english=load_fasttext_model("/home/nitesh/wordembeddings/data/wiki.en.bin")
spanish=load_fasttext_model("/home/nitesh/wordembeddings/data/wiki.es.bin")


model=Disriminator(params)