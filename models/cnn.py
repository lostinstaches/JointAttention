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
        self.efficientnet.eval().to(torch.device("cuda"))

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

    def predict(self, x):
        """Prediction function to generate one-hot encoded output"""
        x_pred = self.forward(x)
        x_final = torch.zeros(self.output_size)
        x_final[torch.argmax(x_pred)] = 1
        return x_final

    def loss(self, x, y):
        """Calculates the loss given input x and true labels y"""
        y_pred = self.forward(x)
        y_true = y
        return self.criterion_gaze(y_pred, y_true)
