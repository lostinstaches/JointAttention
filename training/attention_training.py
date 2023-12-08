import argparse
import glob
import torch
import pandas as pd
print(torch.__version__)
print(torch.cuda.is_available())
from torch import nn
from torch.utils.data import DataLoader
# from data_loader import AttentionDataset
from sklearn.metrics import multilabel_confusion_matrix
import os
from models.mlp import MLP
from models.cnn import CNN
from models.mlp_cnn import MLP_CNN

from sklearn.metrics import precision_score

import random
import numpy as np

import pickle
import io
import copy
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import PIL.Image
from torchvision.transforms import ToTensor
# from modelli import MLP, CNN, MLP_CNN


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
    model = CNN(output_size=7).to('cpu')
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)

    # Pre-Training embeddings
    train(model, optimizer, data_loader['train'], data_loader['val'], opt, log_writer['train'], log_writer['val'],
          model_name)

    # Saving
    torch.save(model.state_dict(), paths['checkpoint_cnn'])

    # Loading
    model.load_state_dict(torch.load(paths['checkpoint_cnn']))
    model.eval()
    return model

def model_mlp_run(data_loader, log_writer, opt, paths, model_name):
    # define the model
    model = MLP(input_size=10, output_size=7).to('cpu')
    # model = model.to('cpu')
    # Define the loss function and optimizer

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)

    # Pre-Training embeddings
    train(model, optimizer, data_loader['train'], data_loader['val'], opt, log_writer['train'], log_writer['val'],
          model_name)

    # Saving
    torch.save(model.state_dict(), paths['checkpoint_mlp'])

    # Loading
    model.load_state_dict(torch.load(paths['checkpoint_mlp']))
    model.eval()

    return model


def model_run(data_loader, log_writer, opt, paths, model_name, model_mlp, model_cnn):
    # define the model
    # n_features = 24
    # n_state = 3
    # n_hidden_size_dec = 10
    # n_features_shape = N_FEATURES
    model = MLP_CNN(model_mlp, model_cnn, output_size=7).to('cpu')

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)

    # Pre-Training embeddings
    train(model, optimizer, data_loader['train'], data_loader['val'], opt, log_writer['train'], log_writer['val'],
          model_name)

    # Saving
    torch.save(model.state_dict(), paths['checkpoint_full'])

    # Plotting and logging stuff
    # print('Final Test acc: %.4f' % (evaluate(model, test_dataloader, gpu_id=opt.gpu_id)))
    # plot
    # str_epochs = [str(i) for i in range(1, opt.epochs + 1)]
    # plot(str_epochs, train_mean_losses, ylabel='Loss', name='training-loss')
    # plot(str_epochs, val_mean_losses, ylabel='Loss', name='validation-loss')

    # Loading
    model.load_state_dict(torch.load(paths['checkpoint_full']))
    model.eval()

    # Memory training
    # final_model = Model(model.encoder_shape, model.decoder_shape, model.encoder_position, model.decoder_position)
    # train_memory(final_model, data_loader['mem'], data_loader['dev'], opt, log_writer['train_mem'], log_writer['val_mem'])
    # torch.save(final_model.state_dict(), paths['checkpoint_full'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='data.pkl',
                        help="Path to data.")
    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-learning_rate', type=float, default=.001)
    parser.add_argument('-l2_decay', type=float, default=0.)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-gpu_id', type=int, default=0)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-model', type=str, default='baseline_memory')

    opt = parser.parse_args()
    sonpath = os.path.dirname(os.path.realpath(__file__))

    checkpoint_path = os.path.join(sonpath, 'weights', 'checkpoint' + str(3) + '.pt')
    checkpoint_cnn = os.path.join(sonpath, 'weights', 'checkpoint' + str(3) + '_cnn.pt')
    checkpoint_path_final = os.path.join(sonpath, 'weights', 'checkpoint' + str(3) + '_final.pt')

    configure_seed(opt.seed)
    # configure_device(opt.gpu_id)

    print("Fixing seed at " + str(opt.seed))
    fix_all_seeds(opt.seed)

    paths = {
        'checkpoint_mlp': checkpoint_path,
        'checkpoint_cnn': checkpoint_cnn,
        'checkpoint_full': checkpoint_path_final
    }

    print("Loading data...")

    # Load data to memory

    base_dir = '../data/pickles/'

    # Find the latest directory
    list_of_dirs = glob.glob(base_dir + '*/')
    print(list_of_dirs)
    latest_dir = max(list_of_dirs, key=os.path.getmtime)

    train_dataset = load_data(os.path.join(latest_dir, 'train.pkl'))
    test_dataset = load_data(os.path.join(latest_dir, 'test.pkl'))
    validation_dataset = load_data(os.path.join(latest_dir, 'val.pkl'))

    data_loader = {
        'train': DataLoader(train_dataset, batch_size=30, shuffle=True),
        'test': DataLoader(test_dataset, batch_size=1),
        'val': DataLoader(validation_dataset, batch_size=1, shuffle=True),
    }

    from datetime import datetime
    now = datetime.now()
    #
    model_name = 'CNN'
    log_writer = {
        'train': SummaryWriter(log_dir='./run_new/' + model_name + '/train/' + now.strftime("%Y-%m-%d-%H-%M-%S"),
                               comment='train' + '_' + model_name),
        'test': SummaryWriter(log_dir='./run_new/' + model_name + '/test/' + now.strftime("%Y-%m-%d-%H-%M-%S"),
                              comment='test' + '_' + model_name),
        'val': SummaryWriter(log_dir='./run_new/' + model_name + '/val/' + now.strftime("%Y-%m-%d-%H-%M-%S"),
                             comment='val' + '_' + model_name),
    }

    if model_name == 'MLP':
        opt.startiter = 0
        model_mlp_run(data_loader, log_writer, opt, paths, model_name)
    elif model_name == 'CNN':
        opt.startiter = 0

        model_cnn_run(data_loader, log_writer, opt, paths, model_name)
    elif model_name == 'ALL':
        opt.startiter = 0
        opt.epochs = 50
        # opt.epochs = 1
        model_train_mlp = model_mlp_run(data_loader, log_writer, opt, paths, 'MLP')
        opt.startiter = 50
        opt.epochs = 15
        # opt.epochs = 1
        model_train_cnn = model_cnn_run(data_loader, log_writer, opt, paths, 'CNN')
        opt.startiter = 75
        # opt.epochs = 1
        opt.epochs = 15
        model_run(data_loader, log_writer, opt, paths, model_name, model_train_mlp, model_train_cnn)
    print("Running ")


def load_data(pickle_file):
    with open(pickle_file, 'rb') as file:
        gaze_data, image_data, labels = pickle.load(file)

    # Convert DataFrame to NumPy array if necessary
    if isinstance(gaze_data, pd.DataFrame):
        gaze_data = gaze_data.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy()

    # Assuming image_data is already a NumPy array as per your print statement
    return TensorDataset(torch.tensor(gaze_data, dtype=torch.float32),
                         torch.tensor(image_data, dtype=torch.float32),
                         torch.tensor(labels, dtype=torch.float32))

def load_data_old(pickle_file):
    with open(pickle_file, 'rb') as file:
        gaze_data, image_data, labels = pickle.load(file)

    # Convert Pandas Series to NumPy array if needed
    if isinstance(gaze_data, pd.Series):
        gaze_data = gaze_data.values
    if isinstance(image_data, pd.Series):
        image_data = image_data.values
    if isinstance(labels, pd.Series):
        labels = labels.values

    return TensorDataset(torch.tensor(gaze_data).float(), torch.tensor(image_data).float(), torch.tensor(labels).float())

def validation_metric(y, y_pred):
    y = y.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    # import pdb; pdb.set_trace()
    acc = np.mean((y[0] == y_pred).all())
    return acc


def precision_metric(y_pred, y):
    y_pred = torch.cat(y_pred)
    print("Shape of concatenated y_pred:", y_pred.shape)
    y_pred = torch.reshape(y_pred, (-1, 7))
    print("Shape of y_pred after reshaping:", y_pred.shape)

    # Concatenate all the true label tensors
    y = torch.cat(y)
    print("Shape of concatenated true labels (y):", y.shape)

    # Convert to numpy arrays for further processing
    y = y.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    # Convert probabilities to predicted class labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y, axis=1)
    print("Predicted labels:", y_pred_labels)
    print("True labels:", y_true_labels)

    # Calculate confusion matrix
    conf_mat = multilabel_confusion_matrix(y_true_labels, y_pred_labels, labels=[0, 1, 2, 3, 4, 5, 6])
    print("Confusion matrix:\n", conf_mat)

    # Calculate precision and accuracy
    sumconf = np.sum(conf_mat, axis=0)

    precision = sumconf[1, 1] / (sumconf[0, 1] + sumconf[1, 1]) if (sumconf[0, 1] + sumconf[1, 1]) > 0 else 0
    accuracy = (sumconf[0, 0] + sumconf[1, 1]) / (sumconf[0, 0] + sumconf[1, 0] + sumconf[1, 1] + sumconf[0, 1])
    return precision, accuracy


from sklearn.metrics import precision_score, accuracy_score

def precision_metric_new(y_pred, y):
    y_pred = torch.cat(y_pred)
    y_pred = torch.reshape(y_pred, (-1, 7))
    y = torch.cat(y)

    y = y.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y, axis=1)

    precision = precision_score(y_true_labels, y_pred_labels, average='macro', labels=np.arange(7), zero_division=0)
    accuracy = accuracy_score(y_true_labels, y_pred_labels)

    return precision, accuracy


def train_batch(x, y, model, optimizer):
    model.train()
    if optimizer is not None:
        optimizer.zero_grad()

    y = y.long()

    loss = model.loss(x, y)

    if optimizer is not None:
        loss.backward()
        optimizer.step()

    return loss.detach()


def train(model, optimizer, train_dataloader, dev_dataloader, opt, train_writer, val_writer, model_name,
          requires_reset=False, viz=True):
    train_mean_losses = []
    val_mean_metrics = []
    # Run the training loop
    for epoch in range(0, opt.epochs):  # 5 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch + 1}')
        train_losses = []
        # import pdb; pdb.set_trace()
        # Iterate over the DataLoader for training data
        for i, sample in enumerate(train_dataloader):
            # print(i)

            gaze, immagine, output = sample
            immagine = immagine.transpose(1,3).transpose(2,3)
            # import pdb; pdb.set_trace()

            gaze, immagine, output = gaze.to('cpu'), immagine.to('cpu'), output.to('cpu')
            if model_name == 'MLP':
                loss = train_batch(gaze, output, model, optimizer)
            elif model_name == 'CNN':
                loss = train_batch(immagine, output, model, optimizer)
            else:
                # if i == 100 or i == 200:
                #     import pdb;pdb.set_trace()
                loss = train_batch([gaze, immagine], output, model, optimizer)

            train_losses.append(loss)
        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % mean_loss)
        train_writer.add_scalar('loss', mean_loss, global_step=opt.startiter + epoch)

        train_mean_losses.append(mean_loss)

    str_epochs = [str(i) for i in range(0, opt.epochs)]
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
        # gaze, immagine, output = gaze.to('cpu'), immagine.to('cpu'), output.to('cpu')

        immagine = immagine.transpose(1, 3).transpose(2, 3)
        if model_name == 'MLP':
            print("MLP")
            # y_pred = model.predict(gaze)
            y_pred = model(gaze)
        elif model_name == 'CNN':
            y_pred = model(immagine) # model.predict(immagine)
        else:
            y_pred = model.predict([gaze, immagine])

        # Convert model output to probabilities

        probabilities = torch.softmax(y_pred, dim=1)

        # Store probabilities and true labels
        y_pred_all.append(probabilities.detach())
        y_true_all.append(output)

        # Calculate and store the metric for this batch
        val_metric = validation_metric(output, probabilities)
        val_metrics.append(val_metric)

        val_metric = validation_metric(output, y_pred)

        y_pred_all.append(y_pred)
        y_true_all.append(output)
        val_metrics.append(val_metric)

    mean_metric_val = torch.tensor(val_metrics).mean().item()

    # import pdb; pdb.set_trace()
    precision, accuracy = precision_metric(y_pred_all, y_true_all)
    print('Epoch %i, Loss: %.4f, Val metric: %.4f' % (epoch, mean_loss, mean_metric_val))
    print('Precision', precision)
    print('Accuracy', accuracy)
    print('Epoch %i, Accuracy: %.4f, Precision: %.4f' % (epoch, np.mean(accuracy), np.mean(precision)))
    val_writer.add_scalar('metric', mean_metric_val, global_step=opt.startiter + epoch)
    val_mean_metrics.append(mean_metric_val)


if __name__ == "__main__":
    main()