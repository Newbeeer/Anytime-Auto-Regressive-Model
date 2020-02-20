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

from error_handling.console_logger import ConsoleLogger
from evaluation.gradient_stats import GradientStats

import torch
import numpy as np
from tqdm import tqdm
import os
import pickle
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

class BaseTrainer(object):

    def __init__(self, device, data_stream, configuration, experiments_path, experiment_name, iterations_to_record=10):
        self._device = device
        self._data_stream = data_stream
        self._configuration = configuration
        self._experiments_path = experiments_path
        self._experiment_name = experiment_name
        self._iterations_to_record = iterations_to_record

    def train(self):
        ConsoleLogger.status('start epoch: {}'.format(self._configuration['start_epoch']))
        ConsoleLogger.status('num epoch: {}'.format(self._configuration['num_epochs']))

        for epoch in range(self._configuration['start_epoch'], self._configuration['num_epochs']):
            with tqdm(self._data_stream.training_loader) as train_bar:
                train_res_recon_error = list() # FIXME: record as a global metric
                train_res_perplexity = list() # FIXME: record as a global metric
                train_res_recon_error_index = dict()
                index_cnt = dict()
                iteration = 0
                loss_sum = 0.0
                max_iterations_number = len(train_bar)
                iterations = list(np.arange(max_iterations_number, step=(max_iterations_number / self._iterations_to_record) - 1, dtype=int))

                for data in train_bar:
                    if len(data['one_hot']) == 1:
                        continue
                    if self._configuration['decoder_type'] == 'deconvolutional':
                        _, loss_res = self.iterate_deconv(data, epoch, iteration, iterations, train_bar, eval=False, return_loss=True)
                        loss_sum += loss_res
                        iteration += 1
                print("Average loss per iteration:", loss_sum / iteration)
                self.save(epoch)
    def eval(self):
        ConsoleLogger.status('start epoch: {}'.format(self._configuration['start_epoch']))
        ConsoleLogger.status('num epoch: {}'.format(self._configuration['num_epochs']))
        for index in [16, 32, 48, 64]:
            recons_loss = 0.0
            cnt = 0.0
            with tqdm(self._data_stream.training_loader) as train_bar:
                train_res_recon_error = list()  # FIXME: record as a global metric
                train_res_perplexity = list()  # FIXME: record as a global metric
                iteration = 0
                max_iterations_number = len(train_bar)
                iterations = list(
                    np.arange(max_iterations_number, step=(max_iterations_number / self._iterations_to_record) - 1,
                              dtype=int))
                for data in train_bar:
                    if len(data['one_hot']) == 1:
                        continue
                    losses, perplexity_value, sample_index = self.iterate_deconv(data, 0, iteration, iterations, train_bar, eval=True, designate=index)
                    if losses is None or perplexity_value is None:
                        continue
                    train_res_recon_error.append(losses)
                    train_res_perplexity.append(perplexity_value)
                    iteration += 1
                    recons_loss += losses['reconstruction_loss']
                    cnt += len(data['one_hot'])

            print("Index:{},Train_res_recon_error:{}".format(index, recons_loss/cnt))
            self.save_eval(0,
                      **{'train_res_recon_error': train_res_recon_error, 'train_res_perplexity': train_res_perplexity})

    def dump(self):
        ConsoleLogger.status('start epoch: {}'.format(self._configuration['start_epoch']))
        ConsoleLogger.status('num epoch: {}'.format(self._configuration['num_epochs']))
        print('Dumping Codebook')
        name = 'wave_bigger_baseline'
        with tqdm(self._data_stream.training_loader) as train_bar:
            iteration = 0
            max_iterations_number = len(train_bar)
            iterations = list(
                np.arange(max_iterations_number, step=(max_iterations_number / self._iterations_to_record) - 1,
                          dtype=int))
            lst = []
            for data in train_bar:
                if len(data['one_hot']) == 1:
                    continue
                if self._configuration['decoder_type'] == 'deconvolutional':
                    indices = self.iterate_deconv(data, 0, iteration, iterations, train_bar, eval=True).view(len(data['one_hot']), -1)
                    iteration += 1
                    lst.append(indices.cpu())
        lst = torch.cat(lst, 0)
        os.makedirs(os.path.join('./data-bin',name), exist_ok=True)
        torch.save(lst, os.path.join('./data-bin', name, 'train.pt'))
        print('Dumped training Codebook')

        with tqdm(self._data_stream.validation_loader) as train_bar:
            iteration = 0
            max_iterations_number = len(train_bar)
            iterations = list(
                np.arange(max_iterations_number, step=(max_iterations_number / self._iterations_to_record) - 1,
                          dtype=int))
            lst = []
            for data in train_bar:
                if len(data['one_hot']) == 1:
                    continue
                if self._configuration['decoder_type'] == 'deconvolutional':
                    indices = self.iterate_deconv(data, 0, iteration, iterations, train_bar, eval=True).view(len(data['one_hot']), -1)
                    iteration += 1
                    lst.append(indices.cpu())
        lst = torch.cat(lst, 0)
        torch.save(lst,  os.path.join('./data-bin', name, 'valid.pt'))
        print('Dumped validation Codebook')

    def fetch(self):
        ConsoleLogger.status('Begin fetch...')
        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 40,
                 }
        indices = torch.LongTensor(np.load('indices_bigger_baseline.npy')).to(self._device)
        idx = [32,128,512]
        title = [0.0625,0.25,1.0]
        for i in range(len(indices)):

            fig, axs = plt.subplots(len(idx), 1, figsize=(35, 30), sharex=True)

            for j in range(len(idx)):
                axs[j].set_title('Fraction of Dimensions:' + str(title[j]), font1)
                x = self._model.indices_fetch(indices[i][:idx[j]].unsqueeze(0))[0, 0].detach().cpu().numpy()
                axs[j].plot(np.arange(len(x)), x)
                write("./audio_baseline/sample_" + str(i) + "_" + str(title[j]) + ".wav", 16000, x)

            plt.savefig('./img_baseline/sample_' + str(i), bbox_inches='tight', pad_inches=0)
            plt.clf()



    def _record_codebook_stats(self, iteration, iterations, vq,
        concatenated_quantized, encoding_indices, speaker_id, epoch):

        if not self._configuration['record_codebook_stats'] or iteration not in iterations:
            return

        embedding = vq.embedding.weight.data.cpu().detach().numpy()
        codebook_stats_entry = {
            'concatenated_quantized': concatenated_quantized.detach().cpu().numpy(),
            'embedding': embedding,
            'n_embedding': embedding.shape[0],
            'encoding_indices': encoding_indices.detach().cpu().numpy(),
            'speaker_ids': speaker_id.to(self._device).detach().cpu().numpy(),
            'batch_size': self._data_stream.training_batch_size
        }
        codebook_stats_entry_path = self._experiments_path + os.sep + \
            self._experiment_name + '_' + str(epoch + 1) + '_' + \
            str(iteration) + '_codebook-stats.pickle'
        with open(codebook_stats_entry_path, 'wb') as file:
            pickle.dump(codebook_stats_entry, file)

    def _record_gradient_stats(self, modules, iteration, iterations, epoch):

        if not self._configuration['record_codebook_stats'] or iteration not in iterations:
            return

        gradient_stats_entry = {
            name: GradientStats.build_gradient_entry(module.named_parameters()) \
            for name, module in modules.items()
        }

        gradient_stats_entry_path = self._experiments_path + os.sep + self._experiment_name + '_' + str(epoch + 1) + '_' + str(iteration) + '_gradient-stats.pickle'
        with open(gradient_stats_entry_path, 'wb') as file:
            pickle.dump(gradient_stats_entry, file)

    def iterate_deconv(self, data, epoch, iteration, iterations, train_bar, eval, designate, return_loss):
        raise NotImplementedError
    def iterate_wavenet(self, data, epoch, iteration, iterations, train_bar, eval, designate):
        raise NotImplementedError

    def save(self, epoch):
        raise NotImplementedError

    def save_eval(self, epoch, **kwargs):
        raise NotImplementedError
