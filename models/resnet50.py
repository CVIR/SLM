import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import math
import numpy as np
from torch.autograd import Variable
from models.resnet_dsbn import *
from torch.autograd import Function
import warnings

warnings.filterwarnings("ignore")


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        # Gradient Reversal Layer (GRL)
        return x

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


class AdversarialNetwork(nn.Module):
    """
    Domain Discriminator Network.
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 2),
            nn.Sigmoid()
        )

    def forward(self, x, lambd):
        x_ = GradReverse(lambd)(x)
        y = self.main(x_)
        return y

class Selector_Network(nn.Module):
    """
    Selector Network.
    """
    def __init__(self):        
        super(Selector_Network, self).__init__()
        self.network = models.resnet18(pretrained=True)
        self.network.fc = nn.Linear(in_features=self.network.fc.in_features, out_features=2)

    def forward(self, out_probs):
        w = self.network(out_probs)
        return w

class ResNet_50_Model(nn.Module):
    """
    Feature Extractor and Classifier Network.
    """
    def __init__(self, pretrained=True, dataset_name=None):
        super(ResNet_50_Model, self).__init__()

        self.network = resnet50dsbn(pretrained=True, in_features=0, num_classes=65, num_domains=2)
        in_features = self.network.fc.in_features
        if(dataset_name=='Office31'):
            self.network.fc = nn.Linear(in_features=in_features, out_features=256)
            self.last_layer = nn.Linear(in_features=256, out_features=31)
        elif(dataset_name=='OfficeHome'):
            self.network.fc = nn.Linear(in_features=in_features, out_features=256)
            self.last_layer = nn.Linear(in_features=256, out_features=65)
        elif(dataset_name=='VisDA2017'):
            self.network.fc = nn.Linear(in_features=in_features, out_features=256)
            self.last_layer = nn.Linear(in_features=256, out_features=12)
        elif(dataset_name=='ImageNetCaltech'):
            self.network.fc = nn.Linear(in_features=in_features, out_features=256)
            self.last_layer = nn.Linear(in_features=256, out_features=1000)
        elif(dataset_name=='CaltechImageNet'):
            self.network.fc = nn.Linear(in_features=in_features, out_features=256)
            self.last_layer = nn.Linear(in_features=256, out_features=256)

        for param in self.network.parameters():
            param.requires_grad = True        


    def forward(self, inputs, domain_label):
        avg_pool_out = self.network(inputs, domain_label)
        btnk_probs = self.network.fc(avg_pool_out)
        out_probs = self.last_layer(btnk_probs)

        return btnk_probs, out_probs