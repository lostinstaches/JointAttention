import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from data_loader import AttentionDataset
from sklearn.metrics import multilabel_confusion_matrix
import os

import random
import numpy as np

import pickle
import io
import copy

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import PIL.Image
from torchvision.transforms import ToTensor
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
def model_cnn_run(data_loader, log_writer, opt, paths, model_name):
    
    # define the model
    # n_features = 24
    # n_state = 3
    # n_hidden_size_dec = 10
    # n_features_shape = N_FEATURES
    model = CNN()
    #model = model.to(opt.gpu_id)
    # Define the loss function and optimizer
    
    model = model.to(opt.gpu_id)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)

    # Pre-Training embeddings
    train(model, optimizer, data_loader['train'], data_loader['val'], opt, log_writer['train'], log_writer['val'], model_name)

    #Saving
    torch.save(model.state_dict(), paths['checkpoint_cnn'])
    
    #Loading
    model.load_state_dict(torch.load(paths['checkpoint_cnn']))
    model.eval()
    return model
    

def model_mlp_run(data_loader, log_writer, opt, paths, model_name):
    
    # define the model
    # n_features = 24
    # n_state = 3
    # n_hidden_size_dec = 10
    # n_features_shape = N_FEATURES
    model = MLP()
    #model = model.to(opt.gpu_id)
    # Define the loss function and optimizer
    
    model = model.to(opt.gpu_id)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)

    # Pre-Training embeddings
    train(model, optimizer, data_loader['train'], data_loader['val'], opt, log_writer['train'], log_writer['val'], model_name)

    #Saving
    torch.save(model.state_dict(), paths['checkpoint_mlp'])
    
    #Loading
    model.load_state_dict(torch.load(paths['checkpoint_mlp']))
    model.eval()

    return model

    
def model_run(data_loader, log_writer, opt, paths, model_name, model_mlp, model_cnn):
    
    # define the model
    # n_features = 24
    # n_state = 3
    # n_hidden_size_dec = 10
    # n_features_shape = N_FEATURES
    model = MLP_CNN(model_mlp, model_cnn)
    
    #model = model.to(opt.gpu_id)
    # Define the loss function and optimizer
    
    model = model.to(opt.gpu_id)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)

    # Pre-Training embeddings
    train(model, optimizer, data_loader['train'], data_loader['val'], opt, log_writer['train'], log_writer['val'], model_name)
 
    #Saving
    torch.save(model.state_dict(), paths['checkpoint_full'])

    # Plotting and logging stuff 
    # print('Final Test acc: %.4f' % (evaluate(model, test_dataloader, gpu_id=opt.gpu_id)))
    # plot
    # str_epochs = [str(i) for i in range(1, opt.epochs + 1)]
    # plot(str_epochs, train_mean_losses, ylabel='Loss', name='training-loss')
   # plot(str_epochs, val_mean_losses, ylabel='Loss', name='validation-loss')
    
    #Loading
    model.load_state_dict(torch.load(paths['checkpoint_full']))
    model.eval()

    # Fill memory
    #model_memory = fill_memory(model, train_dataloader, dev_dataloader)
    
    # Memory training
    #final_model = Model(model.encoder_shape, model.decoder_shape, model.encoder_position, model.decoder_position)  
    #train_memory(final_model, data_loader['mem'], data_loader['dev'], opt, log_writer['train_mem'], log_writer['val_mem'])
    #torch.save(final_model.state_dict(), paths['checkpoint_full'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='data.pkl',
                        help="Path to data.")
    parser.add_argument('-epochs', type=int, default=30)
    parser.add_argument('-learning_rate', type=float, default=.001)
    parser.add_argument('-l2_decay', type=float, default=0.)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-gpu_id', type=int, default=0)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-model', type=str, default='baseline_memory')
    
    opt = parser.parse_args()
    sonpath=os.path.dirname(os.path.realpath(__file__))
    
    checkpoint_path = os.path.join(sonpath, 'weights', 'checkpoint' + str(3) + '.pt')
    checkpoint_cnn = os.path.join(sonpath, 'weights', 'checkpoint' + str(3) + '_cnn.pt')
    checkpoint_path_final = os.path.join(sonpath, 'weights', 'checkpoint' + str(3) + '_final.pt')
    
    configure_seed(opt.seed)
    configure_device(opt.gpu_id)

    print("Fixing seed at "+ str(opt.seed))
    fix_all_seeds(opt.seed)
    
    paths= {
        'checkpoint_mlp': checkpoint_path,
        'checkpoint_cnn': checkpoint_cnn,
        'checkpoint_full': checkpoint_path_final
    }
    
    print("Loading data...")

    # Load data to memory
   
    train_dataset = AttentionDataset(opt.data, 'train')
    test_dataset  = AttentionDataset(opt.data, 'test')
    validation_dataset = AttentionDataset(opt.data, 'val')
    pretrain_dataset = AttentionDataset(opt.data, 'train')
    pretest_dataset  = AttentionDataset(opt.data, 'test')
    

    data_loader = {
        'train': DataLoader(train_dataset, batch_size=50, shuffle=True),
        'pretrain': DataLoader(pretrain_dataset, batch_size=50, shuffle=True),
        'test' : DataLoader(test_dataset, batch_size=1),
        'pretest' : DataLoader(pretest_dataset, batch_size=1),
        'val': DataLoader(validation_dataset, batch_size=1),
        #'preval': DataLoader(pretrain_dataset, batch_size=1, shuffle=True),
    }
 
    from datetime import datetime
    now = datetime.now()
 #
    model_name = 'ALL'
    log_writer = {
       'train': SummaryWriter(log_dir='./run_new/'+model_name+'/train/'+now.strftime("%Y-%m-%d-%H-%M-%S"), comment='train' + '_' + model_name),
       'test' : SummaryWriter(log_dir='./run_new/'+model_name+'/test/'+now.strftime("%Y-%m-%d-%H-%M-%S"), comment='test'+ '_' + model_name),
       'val': SummaryWriter(log_dir='./run_new/'+model_name+'/val/'+now.strftime("%Y-%m-%d-%H-%M-%S"), comment='val' + '_' + model_name),
    }

    if model_name == 'MLP':
        opt.startiter = 0
        model_mlp_run(data_loader,log_writer, opt, paths, model_name)
    elif model_name == 'CNN':
        opt.startiter = 0
        
        model_cnn_run(data_loader,log_writer, opt, paths, model_name)
    elif model_name == 'ALL':
        opt.startiter = 0
        opt.epochs = 50
        # opt.epochs = 1
        model_train_mlp = model_mlp_run(data_loader,log_writer, opt, paths, 'MLP')
        opt.startiter = 50
        opt.epochs = 15
        # opt.epochs = 1
        model_train_cnn = model_cnn_run(data_loader,log_writer, opt, paths, 'CNN')
        opt.startiter = 75
        # opt.epochs = 1
        opt.epochs = 15
        model_run(data_loader,log_writer, opt, paths, model_name, model_train_mlp, model_train_cnn)
    print("Running ")

def validation_metric(y, y_pred):
    y = y.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    
    # import pdb; pdb.set_trace()
    acc = np.mean((y[0] == y_pred).all())
    return acc
def precision_metric(y_pred, y):
    
    y_pred = torch.cat(y_pred)
    y_pred = torch.reshape(y_pred, (-1,6)) 
    y = torch.cat(y)
    y = y.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    yarg = np.argmax(y, axis = 1)
    ypredarg = np.argmax(y_pred, axis = 1)
    
    confmat = multilabel_confusion_matrix(yarg, ypredarg, labels = [0,1,2,3,4,5])




    # import pdb; pdb.set_trace()
    sumconf = np.sum(confmat, axis=0)
    
    precision = sumconf[1,1]/(sumconf[0,1]+ sumconf[1,1])
    accuracy = (sumconf[0,0]+sumconf[1,1])/(sumconf[0,0]+ sumconf[1,0]+sumconf[1,1]+sumconf[0,1])
    # import pdb; pdb.set_trace()
    
    return precision, accuracy
def train_batch(x, y, model, optimizer):
    model.train()
    if optimizer is not None:
        optimizer.zero_grad()

    loss = model.loss(x, y)

    if optimizer is not None:
        loss.backward()
        optimizer.step()
    
    return loss.detach()

def train(model, optimizer, train_dataloader, dev_dataloader, opt, train_writer, val_writer, model_name, requires_reset=False, viz = True):
    train_mean_losses = []
    val_mean_metrics = []
    # Run the training loop
    for epoch in range(0, opt.epochs): # 5 epochs at maximum
        
        # Print epoch
        print(f'Starting epoch {epoch+1}')
        train_losses = []
        # import pdb; pdb.set_trace()
        # Iterate over the DataLoader for training data
        for i, sample in enumerate(train_dataloader):
            # print(i)

            gaze, immagine, output = sample
            # import pdb; pdb.set_trace()

            gaze, immagine, output = gaze.to(opt.gpu_id), immagine.to(opt.gpu_id), output.to(opt.gpu_id)
            if model_name == 'MLP':
                loss = train_batch(gaze, output, model,  optimizer)
            elif model_name == 'CNN':
                loss = train_batch(immagine, output, model,  optimizer)
            else:
                # if i == 100 or i == 200:
                #     import pdb;pdb.set_trace()
                loss = train_batch([gaze, immagine], output, model,  optimizer)
            
            train_losses.append(loss)
        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % mean_loss)
        train_writer.add_scalar('loss', mean_loss, global_step=opt.startiter + epoch)
        
        
        train_mean_losses.append(mean_loss)
    
    str_epochs = [str(i) for i in range(0, opt.epochs )]
    plot(str_epochs, train_mean_losses, ylabel='Loss', name='training-loss')
#    # Process is complete.
    print('Training process has finished.')
    model.eval()

    # import pdb; pdb.set_trace()
    # Validation loop
    val_metrics = []
    y_pred_all = []
    y_true_all = []
   
    for i, sample in enumerate(dev_dataloader):
        gaze, immagine, output = sample
        gaze, immagine, output = gaze.to(opt.gpu_id), immagine.to(opt.gpu_id), output.to(opt.gpu_id)
        if model_name == 'MLP':
            y_pred = model.predict(gaze)
    
        elif model_name == 'CNN':
            y_pred = model.predict(immagine)
        else:
            y_pred = model.predict([gaze,immagine])

        val_metric = validation_metric(output, y_pred)

        y_pred_all.append(y_pred)
        y_true_all.append(output)
        val_metrics.append(val_metric)

        # if i==0 and ii % 40 == 0:
        #     visual_debug(y, y_pred, ii+opt.startiter)
        #     visual_debug_x(x, ii+opt.startiter)
        #     pass   
        
    mean_metric_val = torch.tensor(val_metrics).mean().item()
    
    # import pdb; pdb.set_trace()
    precision, accuracy = precision_metric(y_pred_all,y_true_all)
    print('Epoch %i, Loss: %.4f, Val metric: %.4f' % (epoch, mean_loss, mean_metric_val))
    print('Precision', precision)
    print('Accuracy', accuracy)
    print('Epoch %i, Accuracy: %.4f, Precision: %.4f' % (epoch, np.mean(accuracy), np.mean(precision)))
    val_writer.add_scalar('metric', mean_metric_val, global_step=opt.startiter + epoch)
    val_mean_metrics.append(mean_metric_val)


if __name__ == "__main__":
    main()