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

from experiments.base_trainer import BaseTrainer

import torch
from torch import nn
import torch.optim as optim
import os
import torch.nn.functional as F

class ConvolutionalTrainer(BaseTrainer):

    def __init__(self, device, data_stream, configuration, experiments_path, experiment_name, **kwargs):
        super().__init__(device, data_stream, configuration, experiments_path, experiment_name)

        self._model = kwargs.get('model', None)
        self._criterion = kwargs.get('criterion', nn.MSELoss())
        self._optimizer = kwargs.get('optimizer',
            optim.Adam(self._model.parameters(), lr=configuration['learning_rate'], amsgrad=True))

    def iterate_wavenet(self, data, epoch, iteration, iterations, train_bar, eval=False,designate=None):
        source = data['input_features'].to(self._device)
        speaker_id = data['speaker_id'].to(self._device)
        #target = data['output_features'].to(self._device).permute(0, 2, 1).contiguous().float()
        one_hot = data['one_hot'].to(self._device)
        self._optimizer.zero_grad()
        #print("Souce:{}, one-hot:{}".format(source.size(), one_hot.size()))
        reconstructed_x, vq_loss, losses, perplexity, encoding_indices, concatenated_quantized, sample_index = self._model(source, one_hot.squeeze(), speaker_id, eval=eval, designate=designate)
        reconstructed_x = reconstructed_x.squeeze().permute(0,2,1)
        reconstructed_x = reconstructed_x.reshape(reconstructed_x.size(0) * reconstructed_x.size(1),reconstructed_x.size(2))
        one_hot = one_hot.squeeze().permute(0,2,1)
        target = torch.argmax(one_hot,2).long().view(-1)
        reconstruction_loss = self._criterion(reconstructed_x, target)

        loss = vq_loss + reconstruction_loss
        losses['reconstruction_loss'] = reconstruction_loss.item()
        losses['loss'] = loss.item()

        self._record_codebook_stats(iteration, iterations, self._model.vq,
            concatenated_quantized, encoding_indices, data['speaker_id'], epoch)

        self._record_gradient_stats({'model': self._model, 'encoder': self._model.encoder,
            'vq': self._model.vq, 'decoder': self._model.decoder}, iteration, iterations, epoch)
        if not eval:
            loss.backward()
            self._optimizer.step()

        perplexity_value = perplexity.item()
        train_bar.set_description('Epoch {}: loss {:.4f} perplexity {:.3f}'.format(
            epoch + 1, losses['loss'], perplexity_value))

        return losses, perplexity_value, sample_index

    def iterate_deconv(self, data, epoch, iteration, iterations, train_bar, eval=False,designate=None,return_loss = False):
        wav = data['preprocessed_audio'].to(self._device).squeeze()[:,:15360].unsqueeze(1).permute(0,2,1).contiguous().float()
        target = data['preprocessed_audio'].to(self._device).squeeze()[:, :15360].unsqueeze(1).contiguous().float()
        speaker_id = data['speaker_id'].to(self._device)
        self._optimizer.zero_grad()
        reconstructed_x, z_e_x, z_q_x, index, indices = self._model(wav, self._data_stream.speaker_dic, speaker_id, full=self._configuration['full'], designate=designate)
        reconstruction_loss = F.mse_loss(reconstructed_x, target)
        loss_vq = F.mse_loss(z_q_x[:, :index], z_e_x[:, :index].detach())
        loss_commit = F.mse_loss(z_e_x[:, :index], z_q_x[:, :index].detach())

        loss = reconstruction_loss + loss_vq + 0.2 * loss_commit
        if not eval:
            loss.backward()
            self._optimizer.step()

        train_bar.set_description('Epoch {}: loss recons {:.4f}, loss vq:{:.4f}'.format(
            epoch + 1, reconstruction_loss, loss_vq))
        if return_loss:
            return indices, reconstruction_loss.item()
        else:
            return indices

    def save(self, epoch):
        torch.save({
            'experiment_name': self._experiment_name,
            'epoch': epoch + 1,
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            },
            os.path.join(self._experiments_path, '{}_{}_checkpoint.pth'.format(
                self._experiment_name, epoch + 1))
        )

    def save_eval(self, epoch, **kwargs):
        torch.save({
            'experiment_name': self._experiment_name,
            'epoch': epoch + 1,
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'train_res_recon_error': kwargs.get('train_res_recon_error', -1),
            'train_res_perplexity': kwargs.get('train_res_perplexity', -1)},
            os.path.join(self._experiments_path, '{}_{}_eval_checkpoint.pth'.format(
                self._experiment_name, epoch + 1))
        )
