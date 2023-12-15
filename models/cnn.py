import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

class CNN(nn.Module):
    """
    Convolutional Neural Network based on EfficientNet and additional layers.
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.criterion_gaze = nn.CrossEntropyLoss()
        self.efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        # self.efficientnet.eval().to(torch.device("cuda"))
        self.efficientnet.eval().to('cpu')

        self.avg_pool = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000, 500),
            nn.Softmax(dim=1),
            nn.Linear(500, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        """Forward pass through the CNN"""
        pre_x = self.efficientnet(x)
        pre_x_new = pre_x.detach()
        y = self.avg_pool(pre_x_new)
        return y

    def extract_features(self, x):
        # Pass input through EfficientNet
        features = self.efficientnet(x)
        features_detached = features.detach()  # Detach features to prevent gradients

        # Process features through all avg_pool layers except the last one
        for layer in self.avg_pool[:-1]:
            features_detached = layer(features_detached)
        return features_detached

    def predict(self, x):
        """Prediction function to generate one-hot encoded output"""
        # x_pred = self.forward(x)
        # x_final = torch.zeros(self.output_size)
        # x_final[torch.argmax(x_pred)] = 1
        # return x_final
        x_pred = self.forward(x)
        # Get the index of the max logit.
        # torch.argmax will return the index of the maximum value in each row of x_pred
        predicted_classes = torch.argmax(x_pred, dim=1)

        return predicted_classes

    def loss(self, x, y):
        """Calculates the loss given input x and true labels y"""
        y_pred = self.forward(x)
        y_true = y
        y_true = y_true.squeeze(1)
        return self.criterion_gaze(y_pred, y_true)
