import torch
from torch import nn

class MLP_CNN(nn.Module):
    """
    Combined Multilayer Perceptron and Convolutional Neural Network.
    """
    def __init__(self, model_mlp, model_cnn, output_size):
        super().__init__()
        self.output_size = output_size
        self.criterion_all = nn.CrossEntropyLoss()
        self.model_mlp = model_mlp
        self.model_cnn = model_cnn
        self.avg_pool = nn.Linear(12, output_size)

    def forward(self, x):
        """
        Forward pass through both MLP and CNN, combining their outputs.
        """
        gaze, image = x
        feat_image = self.model_cnn(image)
        feat_gaze = self.model_mlp(gaze)
        total_x = torch.cat([feat_image, feat_gaze], 1)
        y = self.avg_pool(total_x)
        return y

    def predict(self, x):
        """
        Prediction function to generate one-hot encoded output.
        """
        x_pred = self.forward(x)
        x_final = torch.zeros(self.output_size)
        x_final[torch.argmax(x_pred)] = 1
        return x_final

    def loss(self, x, y):
        """
        Calculates the loss given input x and true labels y.
        """
        y_pred = self.forward(x)
        y_true = ((y == 1).nonzero(as_tuple=True)[1])
        return self.criterion_all(y_pred, y_true)
