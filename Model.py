
import torch
from torch import nn


class AEModel(nn.Module):
    # class for mapping attribute vector space to visual feature space
    visual = None
    output = None
    in_dim = None
    h_dim = None
    out_dim = None
    encoder = None
    decoder = None

    def __init__(self, attr_dim, feature_dim):
        super(AEModel, self).__init__()
        self.in_dim = attr_dim
        self.h_dim = feature_dim
        self.out_dim = attr_dim
        self.build_network()

    def build_network(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.h_dim),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(self.h_dim, self.out_dim),
            nn.ReLU(True))

    def forward(self, x):
        self.visual = self.encoder(x)
        self.output = self.decoder(x)
        return self.visual, self.output


class W2AModel(nn.Module):
    # class for finding a mapping between word vectors and attribute vectors
    # motivation: for test classes with unknown attribute vectors
    # this mapping will be used to provide a semantic projection.

    in_dim = None
    h_dim = None
    out_dim = None
    model = None
    attr_pred = None

    def __init__(self, wv_dim, attr_dim):
        super(W2AModel, self).__init__()
        self.in_dim = wv_dim
        self.h_dim = 128
        self.out_dim = attr_dim
        self.build_network()

    def build_network(self):
        self.model = nn.Sequential(
            nn.Linear(self.in_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.out_dim))

    def forward(self, x):
        # self.hidden = self.linear1(x)
        self.attr_pred = self.model(x)
        return self.attr_pred






