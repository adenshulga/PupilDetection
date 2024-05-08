# from torchvision import models
import torch.nn  as nn
# import torch.nn.functional as F
import torch

class MyCNN(nn.Module):

    def __init__(self, num_classes = 2) -> None:
        '''
        conv:
        ouput_size = (input_size - filter_size + 2 x Padding)/stride + 1
        pooling:
        same as conv
        '''
        super(MyCNN, self).__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(3, 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=4, padding=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=4, padding=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),

        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4 * 80, 50),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(50, num_classes)
        )

    def forward(self, x):
        x_before_flatten = self.conv_layers(x)
        x = torch.flatten(x_before_flatten, 1) 
        x = self.fc_layers(x)
        return x, x_before_flatten

def create_model(config):

    model = MyCNN()

    return model