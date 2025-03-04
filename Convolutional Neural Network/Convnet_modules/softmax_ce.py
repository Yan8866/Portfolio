"""
Softmax Cross Entropy Module.  (c) 2021 Georgia Tech

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
import numpy as np

def hello_do_you_copy():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Roger that from softmax_ce.py!")

class SoftmaxCrossEntropy:
    """
    Compute softmax cross-entropy loss given the raw scores from the network.
    """

    def __init__(self):
        self.dx = None
        self.cache = None

    def forward(self, x, y):
        """
        Compute Softmax Cross Entropy Loss
        :param x: raw output of the network: (N, num_classes)
        :param y: labels of samples: (N, )
        :return: computed CE loss of the batch
        """
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        N, _ = x.shape
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        self.cache = (probs, y, N)
        return probs, loss

    def backward(self):
        """
        Compute backward pass of the loss function
        :return:
        """
        probs, y, N = self.cache
        dx = probs.copy()             # creates a copy of the softmax probabilities
        dx[np.arange(N), y] -= 1      #np.arange(N) generates the sample indices, the gradient of softmax cross-entropy w.r.t. logits
        dx /= N                       # normalize the gradient 
        self.dx = dx

