import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

class ChannelAttention1D(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fc(x)
        return x * out


class CNN_Attention(nn.Module):
    """
    Convolutional Neural Network based on EfficientNet with Attention.
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        # Using a larger pre-trained model
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.criterion_gaze = nn.CrossEntropyLoss()

        num_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()  # Remove the final FC layer

        self.channel_attention = ChannelAttention1D(num_channels=num_features)
        # Freeze all layers in EfficientNet
        for param in self.efficientnet.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers
        for layer in list(self.efficientnet.children())[-1:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Possibly add more layers or more complex structures here
        self.avg_pool = nn.Sequential(
            nn.Flatten(),

            nn.Linear(num_features, 1024),  # Increased size
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),  # Dropout to prevent overfitting
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            # Final output layer
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        features = self.efficientnet(x)
        attended_features = self.channel_attention(features)
        y = self.avg_pool(attended_features)
        return y

    def predict(self, x):
        x_pred = self.forward(x)
        predicted_classes = torch.argmax(x_pred, dim=1)
        return predicted_classes

    def extract_features(self, x):
        features = self.efficientnet(x)
        attended_features = self.channel_attention(features)
        return attended_features

    def loss(self, x, y):
        y_pred = self.forward(x)
        y_true = y.squeeze(1)
        return self.criterion_gaze(y_pred, y_true)