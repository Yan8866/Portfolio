"""
2d Convolution Module.  (c) 2021 Georgia Tech

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
    print("Roger that from convolution.py!")

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        (N, C_in, H_in, W_in) = x.shape
        H_out = int((H_in + 2*self.padding - self.kernel_size )/self.stride) + 1 
        W_out = int((W_in + 2*self.padding - self.kernel_size )/self.stride) + 1 

        Z = np.zeros((N, self.out_channels, H_out, W_out ))
        

        X_pad = np.pad(x, ((0,0), (0,0), (self.padding,self.padding), (self.padding, self.padding)), 
                       mode='constant', constant_values = (0,0) )
        
        for i in range(N):
            x_prev_pad = X_pad[i]
            for h in range(H_out):
                vert_start = h * self.stride
                vert_end = vert_start + self.kernel_size

                for w in range(W_out):
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + self.kernel_size

                    for c in range(self.out_channels):
                        x_slice_prev = x_prev_pad[:,vert_start:vert_end, horiz_start:horiz_end]

                        weights = self.weight[c, :, :, :]
                        biases = self.bias[c]
                        Z[i, c, h, w] = np.sum(x_slice_prev * weights) + float(biases)

        out = Z

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        if dout is None:
            raise ValueError(f"{self} received None as input gradient in backward()")
        
        (m, C_out, H_out, W_out) = dout.shape

        self.dx = np.zeros_like(x)
        self.dw = np.zeros_like(self.weight)
        self.db = np.zeros_like(self.bias)

        # Pad x and dx
        X_pad = np.pad(x, ((0,0), (0,0), (self.padding,self.padding), (self.padding, self.padding)), 
                       mode='constant', constant_values = (0,0) )
        dX_pad =np.pad(self.dx, ((0,0), (0,0), (self.padding,self.padding), (self.padding, self.padding)), 
                       mode='constant', constant_values = 0)
        
        for i in range(m):
            x_pad = X_pad[i]
            dx_pad = dX_pad[i]

            for h in range(H_out):                   # loop over vertical axis of the output volume
               for w in range(W_out):                # loop over horizontal axis of the output volume
                   for c in range(C_out):            # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h * self.stride
                    vert_end = vert_start + self.kernel_size
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + self.kernel_size 

                    # Use the corners to define the slice from X_pad
                    x_slice = x_pad[:, vert_start:vert_end, horiz_start:horiz_end]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    dx_pad[:, vert_start:vert_end, horiz_start:horiz_end] += self.weight[c,:,:,:] * dout[i, c, h, w]
                    self.dw[c,:,:,:] += x_slice * dout[i, c, h, w]
                    self.db[c] += dout[i, c, h, w]

            # Set the ith training example's dx to the unpadded dx_pad  
            self.dx[i, :, :, :] = dx_pad[:, self.padding:-self.padding, self.padding:-self.padding]

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
