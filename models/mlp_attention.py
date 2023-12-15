import torch
from torch import nn

class MLP_CNN_Attention(nn.Module):
    """
    Combined Multilayer Perceptron and Convolutional Neural Network.
    """
    def __init__(self, model_mlp, model_cnn_attention, output_size):
        super().__init__()
        self.output_size = output_size
        self.criterion_all = nn.CrossEntropyLoss()
        self.model_mlp = model_mlp
        self.model_cnn_attention = model_cnn_attention
        self.avg_pool = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1600, 1024),  # Adjusted size
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, output_size)
        )

    def forward(self, x):
        """
        Forward pass through both MLP and CNN, combining their outputs.
        """
        gaze, image = x

        # Ensure gaze and image are tensors
        if isinstance(gaze, list):
            gaze = torch.stack(gaze)
        if isinstance(image, list):
            image = torch.stack(image)

        feat_image = self.model_cnn_attention.extract_features(image)
        feat_gaze = self.model_mlp.extract_features(gaze)
        # print("Feat Image Extracted: Shape", feat_image.shape)
        # print("Feat Gaze Extracted: Shape", feat_gaze.shape)
        total_x = torch.cat([feat_image, feat_gaze], 1)
        y = self.avg_pool(total_x)
        return y

    def predict(self, x):
        """
        Prediction function to generate one-hot encoded output.
        """
        x_pred = self.forward(x)
        predicted_classes = torch.argmax(x_pred, dim=1)
        # print("Predicted classes: ", predicted_classes)
        return predicted_classes

    def loss(self, x, y):
        """
        Calculates the loss given input x and true labels y.
        """
        y_pred = self.forward(x)
        y_true = y.squeeze(1)
        return self.criterion_all(y_pred, y_true)
