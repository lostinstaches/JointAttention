import torch
from torch import nn

class MLP(nn.Module):
    '''
    Multilayer Perceptron for processing gaze data.
    '''
    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.criterion_gaze = nn.CrossEntropyLoss()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        """Forward pass through the MLP"""
        return self.layers(x)

    def predict(self, x):
        """Prediction function to generate predicted class labels"""
        with torch.no_grad():
            logits = self.forward(x)
            predicted_classes = torch.argmax(logits, dim=1)
        return predicted_classes

    def loss(self, x, y):
        # """Calculates the loss given input x and true labels y"""
        y_pred = self.forward(x)
        return self.criterion_gaze(y_pred, y.squeeze(dim=1))