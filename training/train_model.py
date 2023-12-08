import glob
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.mlp import MLP
from models.cnn import CNN
from models.mlp_cnn import MLP_CNN
import numpy as np
import pickle
from sklearn.metrics import precision_score
import pandas as pd
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(file_name):
    base_dir = '../data/pickles/'

    # Modify the glob pattern to search for subdirectories within 'pickles'
    list_of_dirs = glob.glob(base_dir + '*/')  # This will include subdirectories
    print(list_of_dirs)  # Debug print to check the directories being listed
    latest_dir = max(list_of_dirs, key=os.path.getmtime)
    file_path = os.path.join(latest_dir, file_name)

    with open(file_path, 'rb') as file:
        gaze_data, images_data, labels_data = pickle.load(file)

    # Convert pandas DataFrame/Series to NumPy array if necessary
    if isinstance(gaze_data, pd.DataFrame) or isinstance(gaze_data, pd.Series):
        gaze_data = gaze_data.values
    if isinstance(images_data, pd.DataFrame) or isinstance(images_data, pd.Series):
        images_data = images_data.values
    if isinstance(labels_data, pd.DataFrame) or isinstance(labels_data, pd.Series):
        labels_data = labels_data.values

    # Convert to PyTorch tensors and create TensorDataset
    gaze_tensor = torch.tensor(gaze_data, dtype=torch.float32)
    images_tensor = torch.tensor(images_data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_data, dtype=torch.float32)

    return TensorDataset(gaze_tensor, images_tensor, labels_tensor)

def train_batch(model, x, y, optimizer):
    """
    Training logic for a single batch.
    """
    model.train()
    optimizer.zero_grad()
    y = y.long()
    loss = model.loss(x, y)
    loss.backward()
    optimizer.step()
    return loss.detach()

def validation_metric(y, y_pred):
    """
    Calculate the validation metric (accuracy in this case).
    """
    y = y.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()


    # import pdb; pdb.set_trace()
    acc = np.mean((y[0] == y_pred).all())
    return acc

    # y_pred_labels = torch.argmax(y_pred, dim=1)
    # correct_predictions = (y_pred_labels == y).sum().item()
    # total_predictions = y.size(0)
    # accuracy = correct_predictions / total_predictions
    # return accuracy

    # y = y.cpu().detach().numpy()
    # y_pred = y_pred.cpu().detach().numpy()
    # return np.mean((y == y_pred).all(axis=1))

def precision_metric(y_pred, y):
    """
    Calculate precision metric for multi-class classification.
    """
    # Convert tensors to numpy arrays
    y_pred = y_pred.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    # Get the predicted labels
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y, axis=1)

    # Calculate precision
    precision = precision_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
    return precision

def train_exp(model, optimizer, train_dataloader, dev_dataloader, epochs, model_type):
    train_mean_losses = []
    val_mean_metrics = []

    for epoch in range(epochs):

        print(f'Starting epoch {epoch+1}')
        train_losses = []

        for i, batch in enumerate(train_dataloader):
            gaze, immagine, output = batch
            gaze, immagine, output = gaze.to(opt.gpu_id), immagine.to(opt.gpu_id), output.to(opt.gpu_id)


def train(model, optimizer, train_dataloader, dev_dataloader, epochs, model_type):
    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Training]"):
            gaze, image, labels = batch
            gaze, image, labels = gaze.to(device), image.to(device), labels.to(device)

            if model_type == 'MLP':
                input_data = gaze
            elif model_type == 'CNN':
                input_data = image
            else:
                input_data = (gaze, image)

            loss = train_batch(model, input_data, labels, optimizer)
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)

        # Validation Phase
        model.eval()
        total_val_correct = 0
        total_val_samples = 0
        for batch in tqdm(dev_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]"):
            gaze, image, labels = batch
            gaze, image, labels = gaze.to(device), image.to(device), labels.to(device)

            if model_type == 'MLP':
                input_data = gaze
            elif model_type == 'CNN':
                input_data = image
            else:
                input_data = (gaze, image)

            with torch.no_grad():
                y_pred = model(input_data)
                total_val_correct += (torch.argmax(y_pred, dim=1) == labels).sum().item()
                total_val_samples += labels.size(0)

        avg_val_accuracy = (total_val_correct / total_val_samples) * 100 if total_val_samples > 0 else 0

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {avg_val_accuracy:.2f}%")


def train_old(model, optimizer, train_dataloader, dev_dataloader, epochs, model_type):
    """
    Training and validation loop.
    """
    train_mean_losses = []
    val_mean_metrics = []

    for epoch in range(epochs):
        train_losses = []


        model.train()
        total_loss = 0

        # Create a progress bar for the training dataloader
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Training]", leave=False)

        for batch in train_progress_bar:
            gaze, image, labels = batch
            gaze, image, labels = gaze.to(device), image.to(device), labels.to(device)

            # Determine the input based on the model type
            if model_type == 'MLP':
                input_data = gaze
            elif model_type == 'CNN':
                input_data = image
            else:
                input_data = (gaze, image)  # For models that require both inputs

            loss = train_batch(model, input_data, labels, optimizer)
            total_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item())
            train_progress_bar.update(1)

        train_progress_bar.close()

        # Validation
        model.eval()
        # total_val_accuracy = 0
        total_val_correct = 0  # Changed: To accumulate correct predictions
        total_val_samples = 0  # Changed: To accumulate total samples
        total_val_precision = 0

        # Create a progress bar for the validation dataloader
        val_progress_bar = tqdm(dev_dataloader, desc=f"Epoch {epoch + 1}/{epochs} [Validation]", leave=False)

        for batch in val_progress_bar:
            gaze, image, labels = batch
            gaze, image, labels = gaze.to(device), image.to(device), labels.to(device)

            # Determine the input for validation
            if model_type == 'MLP':
                input_data = gaze
            elif model_type == 'CNN':
                input_data = image
            else:
                input_data = (gaze, image)

            with torch.no_grad():
                y_pred = model(input_data)
                # val_accuracy = validation_metric(labels, y_pred)
                # precision = precision_metric(y_pred, labels)
                # total_val_accuracy += val_accuracy
                # total_val_precision += precision
                y_pred_labels = torch.argmax(y_pred, dim=1)
                total_val_correct += (y_pred_labels == labels).sum().item()
                total_val_samples += labels.size(0)
                # precision = precision_metric(y_pred, labels)
                # total_val_precision += precision



        avg_loss = total_loss / len(train_dataloader)
        # avg_val_accuracy = total_val_correct / total_val_samples if total_val_samples > 0 else 0
        # avg_val_precision = total_val_precision / len(dev_dataloader)

        avg_val_accuracy = (total_val_correct / total_val_samples) * 100 if total_val_samples > 0 else 0  # Convert to percentage
        avg_val_precision = total_val_precision / len(dev_dataloader)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, "
              f"Val Accuracy: {avg_val_accuracy * 100:.2f}%, Precision: {avg_val_precision:.4f}")


        # avg_loss = total_loss / len(train_dataloader)
        # avg_val_accuracy = total_val_accuracy / len(dev_dataloader)
        # avg_val_precision = total_val_precision / len(dev_dataloader)
        #
        # val_progress_bar.close()
        #
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
        #       f"Val Accuracy: {avg_val_accuracy:.4f}, Precision: {avg_val_precision:.4f}")

def run_training(model_type='MLP', epochs=500, learning_rate=0.005, batch_size=20):
    """
    Main function to run training.
    """
    # Load datasets
    train_dataset = load_dataset('train.pkl')
    val_dataset = load_dataset('val.pkl')
    test_dataset = load_dataset('test.pkl')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Model selection
    if model_type == 'MLP':
        model = MLP(input_size=10, output_size=7).to(device)
    elif model_type == 'CNN':
        model = CNN(output_size=7).to(device)
    else:  # Default to MLP_CNN
        mlp_model = MLP(input_size=10, output_size=7).to(device)
        cnn_model = CNN(output_size=7).to(device)
        model = MLP_CNN(mlp_model, cnn_model, output_size=7).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    train(model, optimizer, train_dataloader, val_dataloader, epochs, model_type)

    # Testing (optional)
    # test(model, test_dataloader)

if __name__ == "__main__":
    run_training()