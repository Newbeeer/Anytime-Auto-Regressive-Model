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

from models.convolutional_encoder import ConvolutionalEncoder,ConvolutionalEncoder_Onehot,ConvolutionalEncoder_V
from models.deconvolutional_decoder import WaveGANGenerator,WaveGANDiscriminator
from models.vector_quantizer import VectorQuantizer,VQEmbedding
from models.vector_quantizer_ema import VectorQuantizerEMA
from error_handling.console_logger import ConsoleLogger
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import os

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class ConvolutionalVQVAE(nn.Module):

    def __init__(self, configuration, device):
        super(ConvolutionalVQVAE, self).__init__()

        self._output_features_filters = configuration['output_features_filters'] * 3 if configuration['augment_output_features'] else configuration['output_features_filters']
        self._output_features_dim = configuration['output_features_dim']
        self._verbose = configuration['verbose']
        self.dim = 512
        self._encoder = WaveGANDiscriminator()

        self._pre_vq_conv = nn.Conv1d(
            in_channels=configuration['num_hiddens'],
            out_channels=configuration['embedding_dim'],
            kernel_size=3,
            padding=1
        )

        if configuration['decay'] > 0.0:
            self._vq = VectorQuantizerEMA(
                num_embeddings=configuration['num_embeddings'],
                embedding_dim=configuration['embedding_dim'],
                commitment_cost=configuration['commitment_cost'],
                decay=configuration['decay'],
                device=device
            )
        else:
            self._vq = VQEmbedding(K=configuration['num_embeddings'], H=60)

        self._decoder = WaveGANGenerator()
        self._device = device
        self._record_codebook_stats = configuration['record_codebook_stats']

    @property
    def vq(self):
        return self._vq

    @property
    def pre_vq_conv(self):
        return self._pre_vq_conv

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def forward(self, x, speaker_dic, speaker_id,full=False, designate=None):

        if self._verbose:
            #print("max:{},min:{}".format(torch.max(x), torch.min(x)))
            ConsoleLogger.status('[ConvVQVAE] _encoder input size: {}'.format(x.size()))
        x = x.permute(0, 2, 1).contiguous().float()
        z = self._encoder(x)
        if self._verbose:
            ConsoleLogger.status('[ConvVQVAE] _encoder output size: {}'.format(z.size()))

        #z = self._pre_vq_conv(z)
        if self._verbose:
            ConsoleLogger.status('[ConvVQVAE] _pre_vq_conv output size: {}'.format(z.size()))

        z_q_x_st, z_q_x, indices = self._vq.straight_through(z)
        area = z.size(1)
        sample_index = area
        if not full:
            sample_index = np.array((np.random.randint(area) + 1))
        if designate != None:
            sample_index = designate
        zero_out = torch.zeros((z_q_x_st.size(0), area - sample_index, z_q_x_st.size(2))).cuda()
        z_q_x_st = torch.cat((z_q_x_st[:, :sample_index], zero_out), dim=1)
        reconstructed_x = self._decoder(z_q_x_st, speaker_dic, speaker_id)
        output_features_size = reconstructed_x.size(2)
        reconstructed_x = reconstructed_x.view(-1, 1, output_features_size)
        return reconstructed_x, z, z_q_x, sample_index, indices


    def indices_fetch(self, indices):

        z = self._vq.indices_fetch(indices)
        if indices.size(1) < self.dim:
            zero_out = torch.zeros((indices.size(0), self.dim - indices.size(1), z.size(2))).cuda()
            z = torch.cat((z, zero_out), dim=1)
        x = self._decoder(z,None,None)
        return x