import torch
from torch.utils.data import DataLoader, TensorDataset
from models.mlp import MLP
from models.cnn import CNN
import numpy as np
from models.mlp_cnn import MLP_CNN
import pickle
from sklearn.metrics import precision_score
import pandas as pd
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(file_path):
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


    # WORKAROUND. Remove when you fix labels
    num_classes = 7
    y = y.clone()  # Clone to avoid modifying the original tensor
    for idx, label in enumerate(y):
        if label < 0:
            y[idx] = 6 # random.randint(0, num_classes - 1)

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
    return np.mean((y == y_pred).all(axis=1))

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

def train(model, optimizer, train_dataloader, dev_dataloader, epochs, model_type):
    """
    Training and validation loop.
    """
    for epoch in range(epochs):
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
        total_val_accuracy = 0
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
                val_accuracy = validation_metric(labels, y_pred)
                precision = precision_metric(y_pred, labels)
                total_val_accuracy += val_accuracy
                total_val_precision += precision

        avg_loss = total_loss / len(train_dataloader)
        avg_val_accuracy = total_val_accuracy / len(dev_dataloader)
        avg_val_precision = total_val_precision / len(dev_dataloader)

        val_progress_bar.close()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
              f"Val Accuracy: {avg_val_accuracy:.4f}, Precision: {avg_val_precision:.4f}")

def run_training(model_type='MLP', epochs=500, learning_rate=0.005, batch_size=20):
    """
    Main function to run training.
    """
    # Load datasets
    train_dataset = load_dataset('../data/pickles/train.pkl')
    val_dataset = load_dataset('../data/pickles/val.pkl')
    test_dataset = load_dataset('../data/pickles/test.pkl')

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