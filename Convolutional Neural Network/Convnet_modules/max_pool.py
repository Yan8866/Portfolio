"""
2d Max Pooling Module.  (c) 2021 Georgia Tech

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
    print("Roger that from max_pool.py!")

class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        N = x.shape[0]
        C = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        H_out = int((H - self.kernel_size )/self.stride + 1)
        W_out = int((W - self.kernel_size )/self.stride + 1)

        out = np.zeros((N, C, H_out, W_out ))

        for i in range(N):
            for h in range(H_out):
               vert_start = h * self.stride
               vert_end = vert_start + self.kernel_size

               for w in range(W_out):
                   horiz_start = w * self.stride
                   horiz_end = horiz_start + self.kernel_size

                   for c in range(C):
                       x_slice = x[i, c, vert_start:vert_end, horiz_start:horiz_end]
                       out[i, c, h, w] = np.max(x_slice)


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        N, C, H, W = x.shape
        self.dx = np.zeros_like(x)

        for i in range(N):
            for c in range(C):
               for h in range(H_out):
                    vert_start = h * self.stride
                    vert_end = vert_start + self.kernel_size

                    for w in range(W_out):
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.kernel_size
                
                        x_slice = x[i, c, vert_start:vert_end, horiz_start:horiz_end]
                        mask = (x_slice == np.max(x_slice))
                        self.dx[i, c, vert_start: vert_end, horiz_start: horiz_end] += mask * dout[i,c,h,w]    
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
