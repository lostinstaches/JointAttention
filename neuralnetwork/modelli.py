import pdb
import torch
from torch import nn
# from matplotlib.pyplot import imread
import numpy as np
import os
import torchvision.transforms as transforms

from efficientnet_pytorch import EfficientNet

mse_loss = torch.nn.MSELoss(reduction='none')

def loss_function(y, y_pred):
    return mse_loss(y, y_pred).sum(axis=1)

class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self):
        super().__init__()
        
        #self.criterion_gaze = nn.MSELoss()
        self.criterion_gaze = nn.CrossEntropyLoss()
        self.layers = nn.Sequential(
            nn.Flatten(),
            #nn.Linear(6, 64),
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)
    def predict(self, x):
        '''Forward pass'''
        # import pdb;pdb.set_trace()
        x_pred = self.layers(x)
        # x_prob = self.softmax(x_pred)
        x_final = torch.zeros(6)
        x_final[torch.argmax(x_pred)] = 1
        return x_final

    def loss (self, x,y):
        y_pred = self.forward(x)
        ytrue =((y == 1).nonzero(as_tuple=True)[1])
        loss = self.criterion_gaze(y_pred, ytrue)
        #loss = self.criterion_gaze(y_pred, y)
        return loss
class CNN(nn.Module):
    '''
    Convolutional Neural Network - based on Efficient Net and WheNet
    '''
    def __init__(self):
        super().__init__()
        self.criterion_immagini = nn.CrossEntropyLoss()
        self.criterion_gaze = nn.MSELoss()
        #self.criterion_gaze = nn.CrossEntropyLoss()
        self.efficientnet =  torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.efficientnet.eval().to(torch.device("cuda"))
        # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
        self.avgpool = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000, 500),
            nn.Softmax(dim=1),
            nn.Linear(500, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        # self.denselay = nn.Linear(15, 6)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        '''Forward pass'''

        # newx = torch.zeros(x.shape[0], x.shape[3], x.shape[1], x.shape[2]).cuda()
        # newx[:,0,:,:] = x[:,:,:,0]
        # newx[:,1,:,:] = x[:,:,:,1]
        # newx[:,2,:,:] = x[:,:,:,2]
        
        # newx_normalize= tfms(newx)
        # 
        prex = self.efficientnet(x)
        #prexnew = prex
        prexnew = prex.detach()
        #import pdb; pdb.set_trace()
        y = self.avgpool(prexnew)
        
        return y
    def predict(self, x):
        '''Forward pass'''
        x_pred = self.forward(x)
        # x_prob = self.softmax(x_pred)
        x_final = torch.zeros(6)
        x_final[torch.argmax(x_pred)] = 1
        return x_final

    def loss (self, x,y):
        y_pred = self.forward(x)
        # import pdb; pdb.set_trace()
        #ytrue =((y == 1).nonzero(as_tuple=True)[1])
        ytrue = y
        loss = self.criterion_gaze(y_pred, ytrue)
        return loss

class MLP_CNN(nn.Module):
    '''
    Multilayer Perceptron and Convolutional Neural Network
    '''
    def __init__(self, model_mlp = 'None', model_cnn ='None'):
        super().__init__()
        self.criterion_all = nn.CrossEntropyLoss()
        self.efficientnet =  model_cnn
        self.mlp =  model_mlp
        # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
        self.avgpool = nn.Linear(12,6)
        # self.denselay = nn.Linear(15, 6)
        # self.softmax = nn.Softmax()

    def forward(self,x):
        '''Forward pass'''

        gaze, immagine = x
        feat_immagine = self.efficientnet(immagine)
        feat_gaze = self.mlp(gaze)
        total_x =  torch.cat([feat_immagine, feat_gaze],1)
        y = self.avgpool(total_x)
        
        return y
    def predict(self, x):
        '''Forward pass'''
        x_pred = self.forward(x)
        # x_prob = self.softmax(x_pred)
        x_final = torch.zeros(6)
        x_final[torch.argmax(x_pred)] = 1
        return x_final

    def loss (self, x,y):
        y_pred = self.forward(x)
        ytrue =((y == 1).nonzero(as_tuple=True)[1])
        
        loss = self.criterion_all(y_pred, ytrue)
        return loss