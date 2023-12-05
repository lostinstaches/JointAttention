import argparse
import optparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from test_dataloader import TestDataset
from data_loader import AttentionDataset, pd
from sklearn.metrics import ConfusionMatrixDisplay, multilabel_confusion_matrix
import os

import random
import numpy as np

import pickle
import io
import copy

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import PIL.Image
from modelli import MLP, CNN, MLP_CNN




def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('{}.pdf'.format(name), bbox_inches='tight')

def fix_all_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def configure_device(gpu_id):
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)

def model_cnn_run(test_dataloader, log_writer, opt, paths, model_name):
    
    model = CNN() 
    model = model.to(opt.gpu_id)
    #optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)
    # Load the trained model in attention training
    model.load_state_dict(torch.load('C:\\Users\\laura\\Documents\\ExpGulbenkian\\demo\\neuralnetwork\\weights\\checkpoint3_cnn.pt'))
    model.eval()
    test(model,test_dataloader['test'], log_writer['test'], opt, model_name)
    
    return model

def model_mlp_run(test_dataloader, log_writer, opt, paths, model_name):

    model = MLP()
    model = model.to(opt.gpu_id)
    
    # Initialize optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)
    # Load the trained model in attention trainingg
    model.load_state_dict(torch.load('C:\\Users\\laura\\Documents\\ExpGulbenkian\\demo\\neuralnetwork\\weights\\checkpoint3.pt'))
    model.eval()
#TEST
    test(model,test_dataloader['test'], log_writer['test'], opt, model_name)
    
    return model

def model_run(test_dataloader, log_writer, opt, paths, model_name):
    model_mlp = MLP()
    model_cnn = CNN()
    model = MLP_CNN(model_mlp, model_cnn)
    model = model.to(opt.gpu_id)
    
    # Initialize optimizer (Adam)
   # optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)
    # Load the trained model in attention training
    model.load_state_dict(torch.load('C:\\Users\\laura\\Documents\\ExpGulbenkian\\demo\\neuralnetwork\\weights\\checkpoint3_final.pt'))
    model.eval()
    test(model,test_dataloader['test'], log_writer['test'], opt, model_name)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='data.pkl',
                        help="Path to data.")
    parser.add_argument('-epochs', type=int, default=80)
    parser.add_argument('-learning_rate', type=float, default=.001)
    parser.add_argument('-l2_decay', type=float, default=0.)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-gpu_id', type=int, default=0)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-model', type=str, default='baseline_memory')

    opt = parser.parse_args()
    sonpath=os.path.dirname(os.path.realpath(__file__))

    checkpoint_path_test_final = os.path.join(sonpath, 'weights', 'checkpoint' + str(3) + '_test_final.pt')
    configure_device(opt.gpu_id)

    paths= {
    'checkpoint_testfinal': checkpoint_path_test_final
    }

    #Load Dataset to memory

    test_dataset  = TestDataset(opt.data, 'test')

    test_dataloader = {
    'test' : DataLoader(test_dataset, batch_size=1),
    }

    
    #Select the model
    model_name='ALL'
    from datetime import datetime
    now = datetime.now()
    log_writer = {
    'test' : SummaryWriter(log_dir='./run_new/'+model_name+'/test/'+now.strftime("%Y-%m-%d-%H-%M-%S"), comment='test'+ '_' + model_name)
    }

    if model_name == 'MLP':
        opt.startiter = 0
        model_mlp_run(test_dataloader,log_writer, opt, paths, model_name)
    elif model_name == 'CNN':
        opt.startiter = 0
        model_cnn_run(test_dataloader,log_writer, opt, paths, model_name)
    elif model_name == 'ALL':
        
        opt.startiter = 0
        opt.epochs = 1
        model_run(test_dataloader,log_writer, opt, paths, model_name)
    print('Running.....')

def validation_metric(y, y_pred):
    y = y.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    
    # import pdb; pdb.set_trace()
    acc = np.mean((y[0] == y_pred).all())
    return acc
def precision_metric(y_pred, y):
    
    y_pred = torch.cat(y_pred)
    y_pred = torch.reshape(y_pred, (-1,6)) 
    #import pdb; pdb.set_trace()
   
    y = torch.cat(y)
    y = y.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    identity_matrix = np.eye(6)#added
    y = identity_matrix[y.astype(int).flatten()]#added
   # print(y)
    #print(y_pred)
    
    yarg = np.argmax(y, axis = 1)
    ypredarg = np.argmax(y_pred, axis = 1)
    #import pdb; pdb.set_trace()

    rutatrue="C:\\Users\\laura\\Documents\\Dados\\Arianna_Alessandro\\CONFMATRIX\\truelabelsE-ALL.txt"
    with open(rutatrue, "w") as archivo:
        for valor in yarg:
            archivo.write(str(valor) + "\n")

    rutapred="C:\\Users\\laura\\Documents\\Dados\\Arianna_Alessandro\\CONFMATRIX\\predlabelsE-ALL.txt"
    with open(rutapred, "w") as archivo2:
        for value in ypredarg:
            archivo2.write(str(value) + "\n")


    confmat = multilabel_confusion_matrix(yarg, ypredarg, labels = [0,1,2,3,4,5])
    sumconf = np.sum(confmat, axis=0)

    print(confmat)
    print(sumconf)
    print(f'TP: {sumconf[1,1]}')
    print(f'FP: {sumconf[0,1]}')
    print(f'TN: {sumconf[0,0]}')
    print(f'FN: {sumconf[1,0]}')
   

    precision = sumconf[1,1]/(sumconf[0,1]+ sumconf[1,1])
    accuracy = (sumconf[0,0]+sumconf[1,1])/(sumconf[0,0]+ sumconf[1,0]+sumconf[1,1]+sumconf[0,1])
    #import pdb; pdb.set_trace()
    
    return precision, accuracy

    
    # return loss.detach()
def test(model, test_dataloader, log_writer, opt, model_name, requires_reset=False, viz = True):
    val_mean_metrics = []
    val_metrics = []
    y_pred_all = []
    y_true_all = []
    
    
    for i, sample in enumerate(test_dataloader):
        gaze, immagine, output = sample
        gaze, immagine, output = gaze.to(opt.gpu_id), immagine.to(opt.gpu_id), output.to(opt.gpu_id)
        if model_name == 'MLP':
            y_pred = model.predict(gaze)    
        elif model_name == 'CNN':
            y_pred = model.predict(immagine)
        else:
            y_pred = model.predict([gaze,immagine])

        test_metric = validation_metric(output, y_pred)
        if output!=-1 and output!=6:
            y_pred_all.append(y_pred)
            y_true_all.append(output)
            val_metrics.append(test_metric)

        mean_metric_val = torch.tensor(val_metrics).mean().item()
   
    precision, accuracy = precision_metric(y_pred_all,y_true_all)
    print(len(y_pred_all))
    print('Precision', precision)
    print('Accuracy', accuracy)
    val_mean_metrics.append(mean_metric_val)
    


if __name__ == "__main__":
    main()