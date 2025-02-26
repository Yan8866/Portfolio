"""
MyModel model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn


def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from my_model.py!")


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        ###########################################################################   
        
        self.block_1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride = 2, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3),
                                   stride=(2, 2))
        )

        # self.shortcut_conv2d = nn.Conv2d(16, 32, kernel_size=1, stride=2, padding=0, bias=False)

        self.block_2 = torch.nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU()
        )

        
        self.block_3 = torch.nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding="same"),
                nn.BatchNorm2d(32),
        )

        self.block_4 = torch.nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU()
        )
        
        self.block_5 = torch.nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding="same"),
                nn.BatchNorm2d(32),
        )

        self.block_6 = torch.nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU()
        )

        self.block_7 = torch.nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding="same"),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride = 1, padding="same"),
                nn.BatchNorm2d(32),
        )

        self.relu = nn.ReLU()
        self.avgpooling = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)


        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1152, 128),  # Assuming CIFAR-10 image size 32x32
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(128, 10)  # Output layer
        )

        # Initialize Weights             
        for m in self.modules():
            if isinstance(m, torch.torch.nn.Conv2d) or isinstance(m, torch.torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        x = self.block_1(x)
        x_shortcut = x
        x = self.block_2(x)
        x = self.block_3(x)
        # print(f"x shape: {x.shape}, x_shortcut shape: {x_shortcut.shape}")
        x.add_(x_shortcut)
        x = self.relu(x)

        x_shortcut = x
        x = self.block_4(x)
        x = self.block_5(x)
        # print(f"x shape: {x.shape}, x_shortcut shape: {x_shortcut.shape}")
        x.add_(x_shortcut)
        x = self.relu(x)

        x_shortcut = x
        x = self.block_6(x)
        x = self.block_7(x)
       # print(f"x shape: {x.shape}, x_shortcut shape: {x_shortcut.shape}")
        x.add_(x_shortcut)
        x = self.relu(x)

        x = self.avgpooling(x)

        # print(f"Feature map shape before flattening: {x.shape}")
        x = x.view(x.size(0), -1) # flatten
        # print(f"Flattened size: {x.shape}")  # Debugging
        outs = self.classifier(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs

