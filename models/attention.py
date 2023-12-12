import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

# class ChannelAttention(nn.Module):
#     def __init__(self, num_channels, reduction_ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
#             nn.ReLU(),
#             nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # Ensure tensor has at least 3 dimensions
#         if x.ndim < 3:
#             raise ValueError("Input tensor to ChannelAttention must have at least 3 dimensions")
#
#         avg_out = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
#         max_out = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
#         out = avg_out + max_out
#         return x * out.unsqueeze(2).unsqueeze(3)


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
        self.criterion_gaze = nn.CrossEntropyLoss()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        num_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Identity()  # Remove the final FC layer

        # self.channel_attention = ChannelAttention(num_channels=num_features)
        self.channel_attention = ChannelAttention1D(num_channels=num_features)

        self.avg_pool = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 500),
            nn.ReLU(),
            nn.Linear(500, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
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

    def loss(self, x, y):
        y_pred = self.forward(x)
        y_true = y.squeeze(1)
        return self.criterion_gaze(y_pred, y_true)