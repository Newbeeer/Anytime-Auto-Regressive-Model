 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2019 Charly Lamothe                                                 #
 # Copyright (C) 2018 Zalando Research                                               #
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

import torch
import torch.nn as nn
from itertools import combinations, product
import numpy as np
import torch.nn.functional as F
import torch
from torch.autograd import Function

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            #print(indices_flatten.size(),inputs.size())
            indices = indices_flatten.view(inputs_size[0], inputs_size[1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)
        # if not ar_:
        #return (codes, indices_flatten, indices)
        # else:
        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]

class VectorQuantizer(nn.Module):
    """
    Inspired from Sonnet implementation of VQ-VAE https://arxiv.org/abs/1711.00937,
    in https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py and
    pytorch implementation of it from zalandoresearch in https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb.

    Implements the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937

    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.
    The output tensor will have the same shape as the input.
    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.
    Args:
        embedding_dim: integer representing the dimensionality of the tensors in the
            quantized space. Inputs to the modules must be in this format as well.
        num_embeddings: integer, the number of vectors in the quantized space.
            commitment_cost: scalar which controls the weighting of the loss terms
            (see equation 4 in the paper - this variable is Beta).
    """
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, device):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = 30
        self._num_embeddings = num_embeddings
        print("VQ embedding dim:{}, num embedding:{}".format(self._embedding_dim,self._num_embeddings))
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        #self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._embedding.weight.data.uniform_(-1, 1)

        self._commitment_cost = commitment_cost
        self._device = device

    def forward(self, inputs, compute_distances_if_possible=True, record_codebook_stats=False,return_index=False, eval=False, designate=None):
        """
        Connects the module to some inputs.

        Args:
            inputs: Tensor, final dimension must be equal to embedding_dim. All other
                leading dimensions will be flattened and treated as a large batch.

        Returns:
            loss: Tensor containing the loss to optimize.
            quantize: Tensor containing the quantized version of the input.
            perplexity: Tensor containing the perplexity of the encodings.
            encodings: Tensor containing the discrete encodings, ie which element
                of the quantized space each input element was mapped to.
            distances
        """

        # Convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(1, 2, 0).contiguous()
        # input_shape = inputs.shape
        # _, time, batch_size = input_shape
        # 3 * 512 * 30
        inputs = inputs.permute(2, 1, 0).contiguous()
        input_shape = inputs.shape
        time, C, batch_size = input_shape

        # Flatten input
        # (batch_size * 64 , embed_dim)
        flat_input = inputs.view(-1, self._embedding_dim)

        # Compute distances between encoded audio frames and embedding vectors
        # (batch_size * 64 , num_embed)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        """
        encoding_indices: Tensor containing the discrete encoding indices, ie
        which element of the quantized space each input element was mapped to.
        """
        # (batch_size * 64 , 1)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # (batch_size * 64 , num_embed)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, dtype=torch.float).to(self._device)
        encodings.scatter_(1, encoding_indices, 1)

        # Compute distances between encoding vectors
        if not self.training and compute_distances_if_possible:
            _encoding_distances = [torch.dist(items[0], items[1], 2).to(self._device) for items in combinations(flat_input, r=2)]
            encoding_distances = torch.tensor(_encoding_distances).to(self._device).view(batch_size, -1)
        else:
            encoding_distances = None

        # Compute distances between embedding vectors
        if not self.training and compute_distances_if_possible:
            _embedding_distances = [torch.dist(items[0], items[1], 2).to(self._device) for items in combinations(self._embedding.weight, r=2)]
            embedding_distances = torch.tensor(_embedding_distances).to(self._device)
        else:
            embedding_distances = None

        # Sample nearest embedding
        if not self.training and compute_distances_if_possible:
            _frames_vs_embedding_distances = [torch.dist(items[0], items[1], 2).to(self._device) for items in product(flat_input, self._embedding.weight.detach())]
            frames_vs_embedding_distances = torch.tensor(_frames_vs_embedding_distances).to(self._device).view(batch_size, C, -1)
        else:
            frames_vs_embedding_distances = None

        # Quantize and unflatten
        # (batch_size * 64 , embed_dim) -> (embed_dim, 64, batch_size)
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # TODO: Check if the more readable self._embedding.weight.index_select(dim=1, index=encoding_indices) works better

        concatenated_quantized = self._embedding.weight[torch.argmin(distances, dim=1).detach().cpu()] if not self.training or record_codebook_stats else None

        area = C
        sample_index = np.array((np.random.randint(area)+1))
        sample_index = np.clip(sample_index, 1, area)
        if eval:
            sample_index = area
        if designate is not None:
            sample_index = designate

        # sample_index = C
        # zero_out = torch.zeros((time, area - sample_index, batch_size)).cuda()
        # quantized = torch.cat((quantized[:, :sample_index], zero_out), dim=1)
        # Losses
        e_latent_loss = F.mse_loss(quantized.detach()[:, :sample_index],inputs[:, :sample_index])
        q_latent_loss = torch.mean((quantized-inputs.detach()) ** 2)
        #e_latent_loss = torch.mean((quantized.detach()[:,:sample_index] - inputs[:,:sample_index])**2)
        #q_latent_loss = torch.mean((quantized[:, :sample_index] - inputs.detach()[:,:sample_index])**2)
        commitment_loss = self._commitment_cost * e_latent_loss
        vq_loss = q_latent_loss + 0. * commitment_loss
        quantized = inputs + (quantized - inputs).detach() # Trick to prevent backpropagation of quantized
        avg_probs = torch.mean(encodings, dim=0)

        """
        The perplexity a useful value to track during training.
        It indicates how many codes are 'active' on average.
        """
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))) # Exponential entropy

        # Convert quantized from BHWC -> BCHW
        if return_index:
            return vq_loss, quantized.permute(2, 1, 0).contiguous(), \
                   perplexity, encodings.view(batch_size, C, -1), \
                   distances.view(batch_size, C, -1), encoding_indices, \
                   {'e_latent_loss': e_latent_loss.item(), 'q_latent_loss': q_latent_loss.item(),
                    'commitment_loss': commitment_loss.item(), 'vq_loss': vq_loss.item()}, \
                   encoding_distances, embedding_distances, frames_vs_embedding_distances, concatenated_quantized, sample_index
        else:
            return vq_loss, quantized.permute(2, 1, 0).contiguous(), \
                   perplexity, encodings.view(batch_size, C, -1), \
                   distances.view(batch_size, C, -1), encoding_indices, \
                   {'e_latent_loss': e_latent_loss.item(), 'q_latent_loss': q_latent_loss.item(),
                    'commitment_loss': commitment_loss.item(), 'vq_loss': vq_loss.item()}, \
                   encoding_distances, embedding_distances, frames_vs_embedding_distances, concatenated_quantized


    @property
    def embedding(self):
        return self._embedding


class VQEmbedding(nn.Module):
    def __init__(self, K, H):
        super().__init__()
        self.scale = 10
        self.embedding = nn.Embedding(K, H)
        self.embedding.weight.data.uniform_(-(1.*self.scale)/K, (1.*self.scale)/K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):

        z_e_x_ = z_e_x.contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.contiguous()
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.contiguous()
        return z_q_x, z_q_x_bar, indices

    def indices_fetch(self, indices):
        indices_flatten = indices.reshape(indices.size(0) * indices.size(1))
        z_q_x_fetch = torch.index_select(self.embedding.weight, dim=0, index=indices_flatten).view(indices.size(0), indices.size(1), 60)

        return z_q_x_fetch
