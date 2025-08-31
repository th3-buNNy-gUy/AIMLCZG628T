import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import warnings
warnings.filterwarnings("ignore")

N_CLASSES = 14

class CWGAN_GP_Generator(nn.Module):
    def __init__(self, feature_noise, feature_size):
        super(CWGAN_GP_Generator, self).__init__()
        self.label_embedding = nn.Embedding(N_CLASSES, 16)

        self.blocks = nn.Sequential(
                                        self._block(feature_noise+16, (feature_noise+16)*16),
                                        self._block((feature_noise+16)*16, (feature_noise+16)*8),
                                        self._block((feature_noise+16)*8, (feature_noise+16)*4),
        )
        self.linear_1 = nn.Linear((feature_noise+16)*4, feature_size)
        self.activation = nn.Identity()

    def _block(self, in_features, out_features):
        return nn.Sequential(
                                nn.Linear(
                                                in_features=in_features,
                                                out_features=out_features,
                                ),
                                nn.BatchNorm1d(out_features),
                                nn.ReLU(),
        )

    def loss(self, fake_scores):
        return -torch.mean(fake_scores)

    def forward(self, x, labels):
        label_embedding = self.label_embedding(labels).view(x.shape[0], -1)
        x = torch.cat((x, label_embedding), dim=1)
        x = self.blocks(x)
        x = self.linear_1(x)
        return self.activation(x)

LABEL_EMBEDDING_G = 64 

class SN_CWGAN_GP_Generator(nn.Module):
    def __init__(self, feature_noise, feature_size):
        super(SN_CWGAN_GP_Generator, self).__init__()
        self.label_embedding = nn.Embedding(N_CLASSES, LABEL_EMBEDDING_G)

        self.blocks = nn.Sequential(
                                        self._block(feature_noise+LABEL_EMBEDDING_G, (feature_noise+LABEL_EMBEDDING_G)*64),
                                        self._block((feature_noise+LABEL_EMBEDDING_G)*64, (feature_noise+LABEL_EMBEDDING_G)*16),
                                        self._block((feature_noise+LABEL_EMBEDDING_G)*16, (feature_noise+LABEL_EMBEDDING_G)*8),
                                        self._block((feature_noise+LABEL_EMBEDDING_G)*8, (feature_noise+LABEL_EMBEDDING_G)*4),
                                        self._block((feature_noise+LABEL_EMBEDDING_G)*4, (feature_noise+LABEL_EMBEDDING_G)*2),
        )
        self.linear_1 = nn.Linear((feature_noise+LABEL_EMBEDDING_G)*2, feature_size)
        self.activation = nn.Identity()

        # self.__init_xavier_paramerters()
        # self.__initialize_weights()

    def _block(self, in_features, out_features):
        return nn.Sequential(
                                nn.utils.spectral_norm(nn.Linear(in_features=in_features, out_features=out_features,)),
                                nn.BatchNorm1d(out_features),
                                nn.LeakyReLU(0.5),
                                # nn.ELU(alpha=0.5),
                                nn.Dropout(0.2)
        )

    def __init_xavier_paramerters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def loss(self, fake_scores):
        return -torch.mean(fake_scores)

    def forward(self, x, labels):
        label_embedding = self.label_embedding(labels).view(x.shape[0], -1)
        x = torch.cat((x, label_embedding), dim=1)
        x = self.blocks(x)
        x = self.linear_1(x)
        return self.activation(x)