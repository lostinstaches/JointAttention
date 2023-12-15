import torch
from torch import nn
import torch.nn.init as init

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
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, output_size)
        )

        self.layers.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the MLP"""
        return self.layers(x)


    def predict(self, x):
        """Prediction function to generate predicted class labels"""
        with torch.no_grad():
            logits = self.forward(x)
            predicted_classes = torch.argmax(logits, dim=1)
        return predicted_classes

    def extract_features(self, x):
        # Process through all layers except the last one
        for layer in self.layers[:-1]:
            x = layer(x)
        return x

    def loss(self, x, y):
        # """Calculates the loss given input x and true labels y"""
        y_pred = self.forward(x)
        return self.criterion_gaze(y_pred, y.squeeze(dim=1))