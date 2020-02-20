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

from models.convolutional_encoder import ConvolutionalEncoder
from models.wavenet_decoder import WaveNetDecoder
from models.vector_quantizer import VectorQuantizer
from models.vector_quantizer_ema import VectorQuantizerEMA

import torch
import torch.nn as nn


class WaveNetVQVAE(nn.Module):
    
    def __init__(self, configuration, speaker_dic, device):
        super(WaveNetVQVAE, self).__init__()
        
        self._encoder = ConvolutionalEncoder(
            in_channels=configuration['input_features_dim'],
            num_hiddens=configuration['num_hiddens'],
            num_residual_layers=configuration['num_residual_layers'],
            num_residual_hiddens=configuration['residual_channels'],
            use_kaiming_normal=configuration['use_kaiming_normal'],
            input_features_type=configuration['input_features_type'],
            features_filters=configuration['input_features_filters'] * 3 if configuration['augment_input_features'] else configuration['input_features_filters'],
            sampling_rate=configuration['sampling_rate'],
            device=device
        )

        self._pre_vq_conv = nn.Conv1d(
            in_channels=configuration['num_hiddens'],
            out_channels=configuration['embedding_dim'],
            kernel_size=1,
            stride=1,
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
        elif configuration['original']:
            print("Use Original VQ")
            self._vq = VectorQuantizer_Original(
                num_embeddings=configuration['num_embeddings'],
                embedding_dim=configuration['embedding_dim'],
                commitment_cost=configuration['commitment_cost'],
                device=device
            )
        else:
            self._vq = VectorQuantizer(
                num_embeddings=configuration['num_embeddings'],
                embedding_dim=configuration['embedding_dim'],
                commitment_cost=configuration['commitment_cost'],
                device=device
            )

        self._decoder = WaveNetDecoder(
            configuration,
            speaker_dic,
            device
        )

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

    def forward(self, x_enc, x_dec, global_condition, eval=False, designate=None):

        x_enc = x_enc.permute(0, 2, 1).contiguous().float()
        z = self._encoder(x_enc)

        z = self._pre_vq_conv(z)

        vq_loss, quantized, perplexity, _, _, encoding_indices, \
            losses, _, _, _, concatenated_quantized, sample_index = self._vq(z, record_codebook_stats=self._record_codebook_stats, return_index=True, eval=eval,designate=designate)

        local_condition = quantized
        local_condition = local_condition.squeeze(-1)
        x_dec = x_dec.squeeze(-1)

        reconstructed_x = self._decoder(x_dec, local_condition, global_condition)
        reconstructed_x = reconstructed_x.unsqueeze(-1)
        x_dec = x_dec.unsqueeze(-1)

        return reconstructed_x, vq_loss, losses, perplexity, encoding_indices, concatenated_quantized, sample_index

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path, configuration, speaker_dic, device):
        model = WaveNetVQVAE(configuration, speaker_dic, device)
        model.load_state_dict(torch.load(path, map_location=device))
        return model
