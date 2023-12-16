import argparse
import glob
import torch
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
import os
from models.mlp import MLP
from models.cnn import CNN
from models.attention import CNN_Attention
from models.mlp_cnn import MLP_CNN
from models.mlp_attention import MLP_CNN_Attention
import random
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
import datetime


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('{}.pdf'.format(name), bbox_inches='tight')

    # Display the plot in a window
    plt.show()


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

def model_cnn_attention_run(data_loader, log_writer, opt, paths, model_name):
    # Instantiate the CNN model with attention
    model = CNN_Attention(output_size=7).to('cpu')

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)


    # Pre-Training embeddings
    train(model, optimizer, data_loader['train'], data_loader['val'], opt, log_writer['train'], log_writer['val'], model_name)

    # Saving the model state
    torch.save(model.state_dict(), paths['checkpoint_cnn'])

    # Loading the model state
    model.load_state_dict(torch.load(paths['checkpoint_cnn']))
    model.eval()

    return model

def model_mlp_run(data_loader, log_writer, opt, paths, model_name):
    # define the model
    model = MLP(input_size=14, output_size=7).to('cpu')
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
    model = MLP_CNN(model_mlp, model_cnn, output_size=7).to('cpu')

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)

    # Pre-Training embeddings
    train(model, optimizer, data_loader['train'], data_loader['val'], opt, log_writer['train'], log_writer['val'],
          model_name)

    # Saving
    torch.save(model.state_dict(), paths['checkpoint_full'])
    # Loading
    model.load_state_dict(torch.load(paths['checkpoint_full']))
    model.eval()

    # Memory training
    # final_model = Model(model.encoder_shape, model.decoder_shape, model.encoder_position, model.decoder_position)
    # train_memory(final_model, data_loader['mem'], data_loader['dev'], opt, log_writer['train_mem'], log_writer['val_mem'])
    # torch.save(final_model.state_dict(), paths['checkpoint_full'])

def model_run_with_attention(data_loader, log_writer, opt, paths, model_name, model_mlp, model_cnn):
    model = MLP_CNN_Attention(model_mlp, model_cnn, output_size=7).to('cpu')

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay)

    train(model, optimizer, data_loader['train'], data_loader['val'], opt, log_writer['train'], log_writer['val'],
          model_name)

    # Saving
    torch.save(model.state_dict(), paths['checkpoint_full_with_attention'])
    # Loading
    model.load_state_dict(torch.load(paths['checkpoint_full_with_attention']))
    model.eval()

def print_label_distribution(dataset, dataset_name):
    # Extract labels from the dataset
    labels = [label.numpy() for _, _, label in dataset]

    # Calculate and print the distribution of labels
    unique, counts = np.unique(np.argmax(labels, axis=1), return_counts=True)
    print(f"Label Distribution in loaded {dataset_name} dataset: {dict(zip(unique, counts))}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='data.pkl',
                        help="Path to data.")
    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-learning_rate', type=float, default=.001)
    parser.add_argument('-l2_decay', type=float, default=0.)
    parser.add_argument('-batch_size', type=int, default=50)
    parser.add_argument('-gpu_id', type=int, default=0)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-model', type=str, default='baseline_memory')

    opt = parser.parse_args()
    sonpath = os.path.dirname(os.path.realpath(__file__))

    checkpoint_path = os.path.join(sonpath, 'weights', 'checkpoint' + str(3) + '.pt')
    checkpoint_cnn = os.path.join(sonpath, 'weights', 'checkpoint' + str(3) + '_cnn.pt')
    checkpoint_path_final = os.path.join(sonpath, 'weights', 'checkpoint' + str(3) + '_final.pt')
    checkpoint_path_final_with_attention = os.path.join(sonpath, 'weights', 'checkpoint' + str(3) + '_final_attention.pt')

    configure_seed(opt.seed)
    # configure_device(opt.gpu_id)

    print("Fixing seed at " + str(opt.seed))
    fix_all_seeds(opt.seed)

    paths = {
        'checkpoint_mlp': checkpoint_path,
        'checkpoint_cnn': checkpoint_cnn,
        'checkpoint_full': checkpoint_path_final,
        'checkpoint_full_with_attention': checkpoint_path_final_with_attention
    }

    print("Loading data...")

    # Load data to memory

    base_dir = '../data/pickles/'

    # Find the latest directory
    list_of_dirs = glob.glob(base_dir + '*/')
    latest_dir = max(list_of_dirs, key=os.path.getmtime)

    train_dataset = load_data(os.path.join(latest_dir, 'train.pkl'))
    test_dataset = load_data(os.path.join(latest_dir, 'test.pkl'))
    validation_dataset = load_data(os.path.join(latest_dir, 'val.pkl'))

    # Print label distributions
    print_label_distribution(train_dataset, "train")
    print_label_distribution(test_dataset, "test")
    print_label_distribution(validation_dataset, "val")

    data_loader = {
        'train': DataLoader(train_dataset, batch_size=40, shuffle=True),
        'test': DataLoader(test_dataset, batch_size=20),
        'val': DataLoader(validation_dataset, batch_size=40),
    }

    from datetime import datetime
    now = datetime.now()
    #
    model_name = 'MLP_ATTENTION'
    # model_name = 'ALL'

    log_writer = {
        'train': SummaryWriter(log_dir='./run_new/' + model_name + '/train/' + now.strftime("%Y-%m-%d-%H-%M-%S"),
                               comment='train' + '_' + model_name),
        'test': SummaryWriter(log_dir='./run_new/' + model_name + '/test/' + now.strftime("%Y-%m-%d-%H-%M-%S"),
                              comment='test' + '_' + model_name),
        'val': SummaryWriter(log_dir='./run_new/' + model_name + '/val/' + now.strftime("%Y-%m-%d-%H-%M-%S"),
                             comment='val' + '_' + model_name),
    }

    dummy_input_mlp = torch.randn(1, 14)  # Replace 14 with the actual input size of the MLP
    dummy_input_cnn = torch.randn(1, 3, 120, 100)  # Replace with the actual input size of the CNN

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
        opt.epochs = 5
        # opt.epochs = 1
        model_train_cnn = model_cnn_run(data_loader, log_writer, opt, paths, 'CNN')
        opt.startiter = 55
        # opt.epochs = 1
        opt.epochs = 5
        model_run(data_loader, log_writer, opt, paths, model_name, model_train_mlp, model_train_cnn)
    elif model_name == 'MLP_ATTENTION':
        opt.startiter = 0
        opt.epochs = 120
        # opt.epochs = 1
        model_train_mlp = model_mlp_run(data_loader, log_writer, opt, paths, 'MLP')
        opt.startiter = 120
        opt.epochs = 20
        # opt.epochs = 1
        model_train_cnn_with_attention = model_cnn_attention_run(data_loader, log_writer, opt, paths, 'Attention')
        opt.startiter = 140
        # opt.epochs = 1
        opt.epochs = 20
        model_run_with_attention(data_loader, log_writer, opt, paths, model_name, model_train_mlp, model_train_cnn_with_attention)
        # model_train_cnn = model_cnn_run(data_loader, log_writer, opt, paths, 'CNN')
        # opt.startiter = 125
        # # opt.epochs = 1
        # opt.epochs = 25
        # model_run_with_attention(data_loader, log_writer, opt, paths, model_name, model_train_mlp, model_train_cnn)

        print("Testing")
        model_weights_path = './weights/checkpoint3_final_attention.pt'
        model_mlp = MLP(input_size=14, output_size=7).to('cpu')
        model_cnn_attention = CNN_Attention(output_size=7).to('cpu')
        test_accuracy, test_precision = evaluate_model_on_test(model_mlp, model_cnn_attention, data_loader['test'],
                                                               model_weights_path)
        print(f"Test Accuracy: {test_accuracy}, Test Precision: {test_precision}")
    print("Running ")

def load_data(pickle_file):
    with open(pickle_file, 'rb') as file:
        gaze_data, image_data, labels = pickle.load(file)

    # Convert DataFrame to NumPy array if necessary
    if isinstance(gaze_data, pd.DataFrame):
        gaze_data = gaze_data.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy()

    # Correct label distribution calculation
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Label Distribution in {pickle_file}: {dict(zip(unique, counts))}")

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

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

def precision_metric_old(y_pred, y):
    y_pred = torch.cat(y_pred)
    y_pred = torch.reshape(y_pred, (-1, 7))

    # Concatenate all the true label tensors
    y = torch.cat(y)

    # Convert to numpy arrays for further processing
    y = y.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    # Convert probabilities to predicted class labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y, axis=1)

    # Calculate confusion matrix
    # conf_mat = multilabel_confusion_matrix(y_true_labels, y_pred_labels, labels=[0, 1, 2, 3, 4, 5, 6])
    conf_mat = confusion_matrix(y_true_labels, y_pred_labels)
    plot_confusion(conf_mat)

    # Calculate precision and accuracy
    sumconf = np.sum(conf_mat, axis=0)

    precision = sumconf[1, 1] / (sumconf[0, 1] + sumconf[1, 1]) if (sumconf[0, 1] + sumconf[1, 1]) > 0 else 0
    accuracy = (sumconf[0, 0] + sumconf[1, 1]) / (sumconf[0, 0] + sumconf[1, 0] + sumconf[1, 1] + sumconf[0, 1])
    return precision, accuracy

def precision_metric(y_pred, y):
    # Concatenate and reshape predicted label tensors
    y_pred = torch.cat(y_pred)
    y_pred = torch.reshape(y_pred, (-1, 7))

    # Concatenate all the true label tensors
    y = torch.cat(y)

    # Convert to numpy arrays for further processing
    y = y.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    # Convert probabilities to predicted class labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = y.squeeze()  # Assuming y is a 2D array of shape (n_samples, 1)

    # Print unique predicted labels
    unique_pred_labels = np.unique(y_pred_labels)
    print(f"Unique Predicted Labels: {unique_pred_labels}")

    # Calculate confusion matrix and plot it
    conf_mat = confusion_matrix(y_true_labels, y_pred_labels)
    plot_confusion(conf_mat)

    # Calculate precision and accuracy
    precision = precision_score(y_true_labels, y_pred_labels, average='macro', labels=np.arange(7), zero_division=0)
    accuracy = accuracy_score(y_true_labels, y_pred_labels)

    return precision, accuracy

def plot_confusion(conf_mat):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    # Get current time for the timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'confusion_matrix_class_{timestamp}.png'

    # Save the figure
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

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
    # print(f"Input x type: {type(x)}, shape: {x[0].shape if isinstance(x, list) else 'Not a list'}")
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
    y_train_pred_all = []
    y_train_true_all = []
    train_accuracies = []
    train_precisions = []

    val_mean_metrics = []
    epoch_accuracies = []
    epoch_precisions = []
    epoch_conf_matrices = []

    # Run the training loop
    for epoch in range(0, opt.epochs):
        # Print epoch
        print(f'Starting epoch {epoch + 1}')
        train_losses = []

        # Iterate over the DataLoader for training data
        for i, sample in enumerate(train_dataloader):
            gaze, immagine, output = sample
            immagine = immagine.transpose(1, 3).transpose(2, 3)
            gaze, immagine, output = gaze.to('cpu'), immagine.to('cpu'), output.to('cpu')
            # print(model_name)
            if model_name == 'MLP':
                loss = train_batch(gaze, output, model, optimizer)
            elif model_name == 'CNN' or model_name == "Attention":
                # print("we entered attention")
                loss = train_batch(immagine, output, model, optimizer)
            elif model_name == "MLP_ATTENTION":
                # print("we entered mlp attention")
                loss = train_batch([gaze, immagine], output, model, optimizer)
            elif model_name == "ALL":
                loss = train_batch([gaze, immagine], output, model, optimizer)
            else:
                print("We should not be here")

                # For calculating training accuracy and precision
            with torch.no_grad():
                if model_name in ['MLP', 'CNN', 'Attention']:
                    y_pred = model(gaze if model_name == 'MLP' else immagine)
                else:
                    y_pred = model([gaze, immagine])

                probabilities = torch.softmax(y_pred, dim=1)
                y_train_pred_all.append(probabilities)
                y_train_true_all.append(output)

            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % mean_loss)
        train_writer.add_scalar('loss', mean_loss, global_step=opt.startiter + epoch)
        train_mean_losses.append(mean_loss)

        # Calculate training accuracy and precision
        train_accuracy, train_precision = precision_metric(y_train_pred_all, y_train_true_all)
        train_accuracies.append(train_accuracy)
        train_precisions.append(train_precision)

        # Print training metrics
        print(
            f'Epoch {epoch + 1}, Training Loss: {mean_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}')

        # Start validation
        model.eval()
        val_metrics = []
        y_pred_all = []
        y_true_all = []

        with torch.no_grad():
            batch_count = 0
            for sample in dev_dataloader:
                gaze, immagine, output = sample
                immagine = immagine.transpose(1, 3).transpose(2, 3)
                gaze, immagine, output = gaze.to('cpu'), immagine.to('cpu'), output.to('cpu')

                # if batch_count == 0:
                #     print(f"Batch shapes - Gaze: {gaze.shape}, Image: {immagine.shape}, Output: {output.shape}")

                if model_name == 'MLP':
                    y_pred = model(gaze)
                    probabilities = torch.softmax(y_pred, dim=1)
                    y_pred_all.append(probabilities.detach())
                elif model_name == 'CNN' or model_name == "Attention":
                    y_pred = model(immagine)
                    probabilities = torch.softmax(y_pred, dim=1)
                    y_pred_all.append(probabilities.detach())
                elif model_name == "MLP_ATTENTION" or model_name == "ALL":
                    # loss = train_batch([gaze, immagine], output, model, optimizer)
                    y_pred = model([gaze, immagine])
                    probabilities = torch.softmax(y_pred, dim=1)
                    y_pred_all.append(probabilities.detach())

                y_true_all.append(output)
                batch_count += 1

            # Calculate overall validation metrics
            precision, accuracy = precision_metric(y_pred_all, y_true_all)
            epoch_accuracies.append(accuracy)
            epoch_precisions.append(precision)
            print(
                f'Epoch {epoch + 1}, Validation - Batches Processed: {batch_count}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}')
            val_writer.add_scalar('metric', accuracy, global_step=opt.startiter + epoch)
            val_mean_metrics.append(accuracy)
            if epoch == opt.epochs - 1:
                y_true_labels = np.concatenate([y.numpy() for y in y_true_all])
                y_pred_labels = np.concatenate([np.argmax(y_pred.numpy(), axis=1) for y_pred in y_pred_all])
                epoch_conf_matrices.append(confusion_matrix(y_true_labels, y_pred_labels))

        # Reset model to training mode
        model.train()

    # Plot training losses
    str_epochs = [str(i) for i in range(0, opt.epochs)]
    plot(str_epochs, train_mean_losses, ylabel='Loss', name='training-loss')

    print('Training process has finished.')
    print('Epoch Accuracies:', epoch_accuracies)
    print('Epoch Precisions:', epoch_precisions)
    print('Confusion Matrix for Last Epoch:', epoch_conf_matrices[-1])


def train_old(model, optimizer, train_dataloader, dev_dataloader, opt, train_writer, val_writer, model_name,
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
            elif model_name == "Attention":
                loss = train_batch(immagine, output, model, optimizer)
            elif model_name == "MLP_ATTENTION":
                loss = train_batch([gaze, immagine], output, model, optimizer)
            else:
                # if i == 100 or i == 200:
                #     import pdb;pdb.set_trace()
                # print("We are in else")
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
        elif model_name == "Attention":
            y_pred = model(immagine)
        elif model_name == "MLP_ATTENTION":
            y_pred = model([gaze, immagine])
        else:
            y_pred = model([gaze, immagine])

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

# def evaluate_model_on_test(model, test_dataloader):
#     model.eval()  # Set the model to evaluation mode
#     y_pred_all = []
#     y_true_all = []
#
#     with torch.no_grad():  # No need to track gradients during evaluation
#         for sample in test_dataloader:
#             gaze, immagine, output = sample
#             immagine = immagine.transpose(1, 3).transpose(2, 3)
#             gaze, immagine, output = gaze.to('cpu'), immagine.to('cpu'), output.to('cpu')
#
#             # Make predictions
#             y_pred = model([gaze, immagine])
#             probabilities = torch.softmax(y_pred, dim=1)
#
#             # Store predictions and true labels
#             y_pred_all.append(probabilities)
#             y_true_all.append(output)
#
#     # Calculate overall metrics for the test set
#     precision, accuracy = precision_metric(y_pred_all, y_true_all)
#     print(f'Test set evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}')
#
#     return accuracy, precision


def evaluate_model_on_test_with_path(model, test_dataloader, model_weights_path):
    # Load the saved model weights
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()  # Set the model to evaluation mode

    y_pred_all = []
    y_true_all = []

    with torch.no_grad():  # No need to track gradients during evaluation
        for sample in test_dataloader:
            gaze, immagine, output = sample
            immagine = immagine.transpose(1, 3).transpose(2, 3)
            gaze, immagine, output = gaze.to('cpu'), immagine.to('cpu'), output.to('cpu')

            # Make predictions
            y_pred = model([gaze, immagine])
            probabilities = torch.softmax(y_pred, dim=1)

            # Store predictions and true labels
            y_pred_all.append(probabilities)
            y_true_all.append(output)

    # Calculate overall metrics for the test set
    precision, accuracy = precision_metric(y_pred_all, y_true_all)
    print(f'Test set evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}')

    return accuracy, precision

def evaluate_model_on_test(model_mlp, model_cnn_attention, test_dataloader, model_weights_path):
    # Load the saved model weights into the MLP_CNN_Attention model
    model = MLP_CNN_Attention(model_mlp, model_cnn_attention, output_size=7).to('cpu')
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()  # Set the model to evaluation mode

    y_pred_all = []
    y_true_all = []

    with torch.no_grad():  # No need to track gradients during evaluation
        for sample in test_dataloader:
            gaze, immagine, output = sample
            immagine = immagine.transpose(1, 3).transpose(2, 3)
            gaze, immagine, output = gaze.to('cpu'), immagine.to('cpu'), output.to('cpu')

            # Make predictions using both gaze and image data
            y_pred = model([gaze, immagine])
            probabilities = torch.softmax(y_pred, dim=1)

            # Store predictions and true labels
            y_pred_all.append(probabilities)
            y_true_all.append(output)

    # Calculate overall metrics for the test set
    precision, accuracy = precision_metric(y_pred_all, y_true_all)
    print(f'Test set evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}')

    return accuracy, precision

def evaluate_on_new_data(test_dataset_dir):
    test_dataset = load_data(os.path.join(test_dataset_dir, 'test.pkl'))
    data_loader = {
        'test': DataLoader(test_dataset, batch_size=40)
    }
    model_weights_path = './weights/checkpoint3_final_attention.pt'
    model_mlp = MLP(input_size=14, output_size=7).to('cpu')
    model_cnn_attention = CNN_Attention(output_size=7).to('cpu')
    test_accuracy, test_precision = evaluate_model_on_test(model_mlp, model_cnn_attention, data_loader['test'],                                                   model_weights_path)
    print(f"Test Accuracy: {test_accuracy}, Test Precision: {test_precision}")

if __name__ == "__main__":
    main()