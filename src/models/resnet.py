import torch
from torch import Tensor
import torch.nn as nn



class ResNet(nn.Module):
    """
    A simple ResNet model for image classification, consisting of multiple convolutional layers 
    followed by residual blocks and fully connected layers.
    
    Attributes:
        conv1 (nn.Conv2d): Initial convolutional layer.
        res1, res2, res3, res4, res5 (ResBlock): Residual blocks.
        pool1, pool2, pool3, pool4, pool5 (nn.MaxPool2d): Max pooling layers.
        flatten (nn.Flatten): Flatten layer to prepare for fully connected layers.
        fc (nn.Sequential): Fully connected layers for classification.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.res1 = ResBlock(64)
        self.pool1 = nn.MaxPool2d(2, 2) # (64, 64, 64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.res2 = ResBlock(128)
        self.pool2 = nn.MaxPool2d(2, 2) # (128, 32, 32)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.res3 = ResBlock(256)
        self.pool3 = nn.MaxPool2d(2, 2) # (256, 16, 16)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.res4 = ResBlock(512)
        self.pool4 = nn.MaxPool2d(2, 2) # (512, 8, 8)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.res5 = ResBlock(1024)
        self.pool5 = nn.MaxPool2d(2, 2) # (1024, 4, 4)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 6)
        )

    
    def forward(self, x: Tensor):
        """
        Forward pass of the ResNet model.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, 29) representing class scores.
        """
        x = self.conv1(x)
        x = self.res1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.res2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.res3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.res4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.res5(x)
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x  # out put shape -> (batch_size, 29)


class ResBlock(nn.Module):
    """
    A Residual Block that applies two convolutional layers with batch normalization 
    and a ReLU activation while adding a skip connection.
    
    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        batch_norm1 (nn.BatchNorm2d): Batch normalization after first convolution.
        conv2 (nn.Conv2d): Second convolutional layer.
        batch_norm2 (nn.BatchNorm2d): Batch normalization after second convolution.
        relu (nn.ReLU): Activation function.
    """
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor):
        """
        Forward pass of the residual block.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).
        
        Returns:
            Tensor: Output tensor after applying convolutions, batch normalization, and ReLU activation.
        """
        sc = x # Skip connection

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = out + sc # Add skip connection
        out = self.relu(out)
        return out