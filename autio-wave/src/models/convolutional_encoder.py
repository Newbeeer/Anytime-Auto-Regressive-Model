 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 #                                                                                   #
 # This file is part of VQ-VAE-Speech.                                               #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

from modules.residual_stack import ResidualStack
from modules.conv1d_builder import Conv1DBuilder
from error_handling.console_logger import ConsoleLogger

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalEncoder(nn.Module):
    
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
        use_kaiming_normal, input_features_type, features_filters, sampling_rate,
        device, verbose=False):

        super(ConvolutionalEncoder, self).__init__()

        """
        2 preprocessing convolution layers with filter length 3
        and residual connections.
        """

        self._conv_1 = Conv1DBuilder.build(
            in_channels=256,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._conv_2 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_3 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2, # timestep * 2
            use_kaiming_normal=use_kaiming_normal,
            padding=2
        )

        """
        2 convolutional layers with length 3 and
        residual connections.
        """

        self._conv_4 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._conv_5 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        4 feedforward ReLu layers with residual connections.
        """

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal
        )
        
        self._input_features_type = input_features_type
        self._features_filters = features_filters
        self._sampling_rate = sampling_rate
        self._device = device
        self._verbose = verbose

    def forward(self, inputs):
        if self._verbose:
            ConsoleLogger.status('inputs size: {}'.format(inputs.size()))

        x_conv_1 = F.relu(self._conv_1(inputs))
        if self._verbose:
            ConsoleLogger.status('x_conv_1 output size: {}'.format(x_conv_1.size()))

        x = F.relu(self._conv_2(x_conv_1)) + x_conv_1
        if self._verbose:
            ConsoleLogger.status('_conv_2 output size: {}'.format(x.size()))
        
        x_conv_3 = F.relu(self._conv_3(x))
        if self._verbose:
            ConsoleLogger.status('_conv_3 output size: {}'.format(x_conv_3.size()))

        x_conv_4 = F.relu(self._conv_4(x_conv_3)) + x_conv_3
        if self._verbose:
            ConsoleLogger.status('_conv_4 output size: {}'.format(x_conv_4.size()))

        x_conv_5 = F.relu(self._conv_5(x_conv_4)) + x_conv_4
        if self._verbose:
            ConsoleLogger.status('x_conv_5 output size: {}'.format(x_conv_5.size()))

        x = self._residual_stack(x_conv_5) + x_conv_5
        if self._verbose:
            ConsoleLogger.status('_residual_stack output size: {}'.format(x.size()))

        return x


class ConvolutionalEncoder_Onehot(nn.Module):


     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                  use_kaiming_normal, input_features_type, features_filters, sampling_rate,
                  device, verbose=False):

         super(ConvolutionalEncoder_Onehot, self).__init__()

         """
         2 preprocessing convolution layers with filter length 3
         and residual connections.
         """

         self._conv_1 = Conv1DBuilder.build(
             in_channels=256,
             out_channels=num_hiddens,
             kernel_size=4,
             use_kaiming_normal=use_kaiming_normal,
             padding=1,
             stride=2
         )

         self._conv_2 = Conv1DBuilder.build(
             in_channels=num_hiddens,
             out_channels=num_hiddens,
             kernel_size=4,
             use_kaiming_normal=use_kaiming_normal,
             padding=1,
             stride=2
         )

         self._conv_3 = Conv1DBuilder.build(
             in_channels=num_hiddens,
             out_channels=num_hiddens,
             kernel_size=4,
             use_kaiming_normal=use_kaiming_normal,
             padding=1,
             stride=2
         )

         self._conv_4 = Conv1DBuilder.build(
             in_channels=num_hiddens,
             out_channels=num_hiddens,
             kernel_size=4,
             use_kaiming_normal=use_kaiming_normal,
             padding=1,
             stride=2
         )

         self._conv_5 = Conv1DBuilder.build(
             in_channels=num_hiddens,
             out_channels=num_hiddens,
             kernel_size=4,
             use_kaiming_normal=use_kaiming_normal,
             padding=1,
             stride=2
         )

         self._conv_6 = Conv1DBuilder.build(
             in_channels=num_hiddens,
             out_channels=num_hiddens,
             kernel_size=4,
             use_kaiming_normal=use_kaiming_normal,
             padding=1,
             stride=2
         )

         self._input_features_type = input_features_type
         self._features_filters = features_filters
         self._sampling_rate = sampling_rate
         self._device = device
         self._verbose = verbose

     def forward(self, inputs):
         if self._verbose:
             ConsoleLogger.status('inputs size: {}'.format(inputs.size()))

         x_conv_1 = F.relu(self._conv_1(inputs))
         if self._verbose:
             ConsoleLogger.status('x_conv_1 output size: {}'.format(x_conv_1.size()))

         x = F.relu(self._conv_2(x_conv_1))
         if self._verbose:
             ConsoleLogger.status('_conv_2 output size: {}'.format(x.size()))

         x_conv_3 = F.relu(self._conv_3(x))
         if self._verbose:
             ConsoleLogger.status('_conv_3 output size: {}'.format(x_conv_3.size()))

         x_conv_4 = F.relu(self._conv_4(x_conv_3))
         if self._verbose:
             ConsoleLogger.status('_conv_4 output size: {}'.format(x_conv_4.size()))

         x_conv_5 = F.relu(self._conv_5(x_conv_4))
         if self._verbose:
             ConsoleLogger.status('x_conv_5 output size: {}'.format(x_conv_5.size()))

         x = F.relu(self._conv_6(x_conv_5))
         if self._verbose:
             ConsoleLogger.status('_residual_stack output size: {}'.format(x.size()))

         return x


class ConvolutionalEncoder_V(nn.Module):

     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                  use_kaiming_normal, input_features_type, features_filters, sampling_rate,
                  device, verbose=False):

         super(ConvolutionalEncoder_V, self).__init__()

         """
         2 preprocessing convolution layers with filter length 3
         and residual connections.
         """

         self._conv_1 = Conv1DBuilder.build(
             in_channels=1,
             out_channels=64,
             kernel_size=4,
             use_kaiming_normal=use_kaiming_normal,
             padding=1,
             stride=2
         )

         self._conv_2 = Conv1DBuilder.build(
             in_channels=64,
             out_channels=128,
             kernel_size=4,
             use_kaiming_normal=use_kaiming_normal,
             padding=1,
             stride=2
         )

         self._conv_3 = Conv1DBuilder.build(
             in_channels=128,
             out_channels=256,
             kernel_size=4,
             use_kaiming_normal=use_kaiming_normal,
             padding=1,
             stride=2
         )

         self._conv_4 = Conv1DBuilder.build(
             in_channels=256,
             out_channels=512,
             kernel_size=4,
             use_kaiming_normal=use_kaiming_normal,
             padding=1,
             stride=2
         )

         self._conv_5 = Conv1DBuilder.build(
             in_channels=512,
             out_channels=768,
             kernel_size=4,
             use_kaiming_normal=use_kaiming_normal,
             padding=1,
             stride=2
         )

         self._conv_6 = Conv1DBuilder.build(
             in_channels=768,
             out_channels=768,
             kernel_size=4,
             use_kaiming_normal=use_kaiming_normal,
             padding=1,
             stride=2
         )


         self._input_features_type = input_features_type
         self._features_filters = features_filters
         self._sampling_rate = sampling_rate
         self._device = device
         self._verbose = verbose

     def forward(self, inputs):
         if self._verbose:
             ConsoleLogger.status('inputs size: {}'.format(inputs.size()))

         x_conv_1 = F.relu(self._conv_1(inputs))
         if self._verbose:
             ConsoleLogger.status('x_conv_1 output size: {}'.format(x_conv_1.size()))

         x_conv_2 = F.relu(self._conv_2(x_conv_1))
         if self._verbose:
             ConsoleLogger.status('_conv_2 output size: {}'.format(x_conv_2.size()))

         x_conv_3 = F.relu(self._conv_3(x_conv_2))
         if self._verbose:
             ConsoleLogger.status('_conv_3 output size: {}'.format(x_conv_3.size()))

         x_conv_4 = F.relu(self._conv_4(x_conv_3))
         if self._verbose:
             ConsoleLogger.status('_conv_4 output size: {}'.format(x_conv_4.size()))

         x_conv_5 = F.relu(self._conv_5(x_conv_4))
         if self._verbose:
             ConsoleLogger.status('x_conv_5 output size: {}'.format(x_conv_5.size()))

         x_conv_6 = self._conv_6(x_conv_5)

         return x_conv_6