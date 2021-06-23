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

from dataset.spectrogram_parser import SpectrogramParser
from dataset.vctk import VCTK
from error_handling.console_logger import ConsoleLogger
from evaluation.alignment_stats import AlignmentStats
from evaluation.embedding_space_stats import EmbeddingSpaceStats

import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import numpy as np
from textwrap import wrap
import seaborn as sns
import textgrid
from tqdm import tqdm
import pickle
import torch
import librosa
from scipy.io.wavfile import write
import seaborn as sns
import pandas as pd

class Evaluator(object):

    def __init__(self, device, model, data_stream, configuration, results_path, experiment_name):
        self._device = device
        self._model = model
        self._data_stream = data_stream
        self._configuration = configuration
        self._vctk = VCTK(self._configuration['data_root'], ratio=self._configuration['train_val_split'])
        self._results_path = results_path
        self._experiment_name = experiment_name

    def evaluate(self, evaluation_options):
        self._model.eval()

        if evaluation_options['plot_exp'] or \
            evaluation_options['plot_quantized_embedding_spaces'] or \
            evaluation_options['plot_distances_histogram']:
            evaluation_entry = self._evaluate_once()

        if evaluation_options['plot_exp']:
            self._compute_comparaison_plot(evaluation_entry)

        if evaluation_options['plot_quantized_embedding_spaces']:
            EmbeddingSpaceStats.compute_and_plot_quantized_embedding_space_projections(
                self._results_path, self._experiment_name, evaluation_entry,
                self._model.vq.embedding, self._data_stream.validation_batch_size
            )

        if evaluation_options['plot_distances_histogram']:
            self._plot_distances_histogram(evaluation_entry)

        #self._test_denormalization(evaluation_entry)

        if evaluation_options['compute_many_to_one_mapping']:
            self._many_to_one_mapping()

        if evaluation_options['compute_alignments'] or \
            evaluation_options['compute_clustering_metrics'] or \
            evaluation_options['compute_groundtruth_average_phonemes_number']:
            alignment_stats = AlignmentStats(
                self._data_stream,
                self._vctk,
                self._configuration,
                self._device,
                self._model,
                self._results_path,
                self._experiment_name,
                evaluation_options['alignment_subset']
            )
        if evaluation_options['compute_alignments']:
            groundtruth_alignments_path = self._results_path + os.sep + \
                'vctk_{}_groundtruth_alignments.pickle'.format(evaluation_options['alignment_subset'])
            if not os.path.isfile(groundtruth_alignments_path):
                alignment_stats.compute_groundtruth_alignments()
                alignment_stats.compute_groundtruth_bigrams_matrix(wo_diag=True)
                alignment_stats.compute_groundtruth_bigrams_matrix(wo_diag=False)
                alignment_stats.compute_groundtruth_phonemes_frequency()
            else:
                ConsoleLogger.status('Groundtruth alignments already exist')

            empirical_alignments_path = self._results_path + os.sep + self._experiment_name + \
                '_vctk_{}_empirical_alignments.pickle'.format(evaluation_options['alignment_subset'])
            if not os.path.isfile(empirical_alignments_path):
                alignment_stats.compute_empirical_alignments()
                alignment_stats.compute_empirical_bigrams_matrix(wo_diag=True)
                alignment_stats.compute_empirical_bigrams_matrix(wo_diag=False)
                alignment_stats.comupte_empirical_encodings_frequency()
            else:
                ConsoleLogger.status('Empirical alignments already exist')

        if evaluation_options['compute_clustering_metrics']:
            alignment_stats.compute_clustering_metrics()

        if evaluation_options['compute_groundtruth_average_phonemes_number']:
            alignment_stats.compute_groundtruth_average_phonemes_number()

    def _evaluate_once(self):
        self._model.eval()
        data = next(iter(self._data_stream.training_loader))
        # sample = 5 sample =1
        sample = 5
        preprocessed_audio = data['preprocessed_audio'].to(self._device)
        valid_originals = data['input_features'].to(self._device)
        speaker_ids = data['speaker_id'].to(self._device)
        target = data['output_features'].to(self._device)
        wav_filename = data['wav_filename']
        wav = data['preprocessed_audio'].to(self._device).squeeze()[:, :15360].unsqueeze(1).contiguous().float()
        shifting_time = data['shifting_time'].to(self._device)
        preprocessed_length = data['preprocessed_length'].to(self._device)
        one_hot = data['one_hot'].to(self._device)
        batch_size = preprocessed_audio.size(0)
        wav_filename = wav_filename[0][sample]
        print("Wav size:",wav.size())
        z = self._model.encoder(wav)
        print("Encoder out size:",z.size())
        valid_reconstructions = list()

        for i in [32,128,512]:
            z_q_x_st, z_q_x, indices = self._model.vq.straight_through(z)
            zero_out = torch.zeros((z_q_x_st.size(0), 512 - i, z_q_x_st.size(2))).cuda()
            z_q_x_st = torch.cat((z_q_x_st[:, :i], zero_out), dim=1)
            valid_reconstructions.append(self._model.decoder(z_q_x_st, self._data_stream.speaker_dic, speaker_ids)[sample])
            torch.cuda.empty_cache()

        return {
            'preprocessed_audio': preprocessed_audio,
            'valid_originals': valid_originals,
            'speaker_ids': speaker_ids,
            'target': target,
            'wav_filename': wav_filename,
            'shifting_time': shifting_time,
            'preprocessed_length': preprocessed_length,
            'batch_size': batch_size,
            'valid_reconstructions': valid_reconstructions,
            'one_hot': one_hot,
            'sample': sample
        }

    def _compute_comparaison_plot(self, evaluation_entry):
        print("entry:",evaluation_entry['wav_filename'])
        sample = evaluation_entry['sample']
        utterence_key = evaluation_entry['wav_filename'].split('/')[-1].replace('.wav', '')
        utterence = self._vctk.utterences[utterence_key].replace('\n', '')
        phonemes_alignment_path = os.sep.join(evaluation_entry['wav_filename'].split('/')[:-3]) \
            + os.sep + 'phonemes' + os.sep + utterence_key.split('_')[0] + os.sep \
            + utterence_key + '.TextGrid'

        speaker = evaluation_entry['speaker_ids']
        ConsoleLogger.status('Original utterence: {}, speaker id:{}'.format(utterence,speaker))

        if self._configuration['verbose']:
            ConsoleLogger.status('utterence: {}'.format(utterence))
        spectrogram_parser = SpectrogramParser()
        preprocessed_audio = evaluation_entry['preprocessed_audio'].detach().cpu()[sample].numpy().squeeze()

        sns.set(style='darkgrid', font_scale=5)
        preprocessed_audio = preprocessed_audio[:15360]
        valid_reconstructions = evaluation_entry['valid_reconstructions']
        fig, axs = plt.subplots(1,len(valid_reconstructions), figsize=(120, 20), sharex=True)
        # Waveform of the original speech signal
        # axs[0].set_title('Waveform of the original speech signal')
        # axs[0].plot(np.arange(len(preprocessed_audio)), preprocessed_audio)
        write("original.wav", 16000, preprocessed_audio)
        # # Spectrogram of the original speech signal
        # axs[1].set_title('Spectrogram of the original speech signal')
        # self._plot_pcolormesh(spectrogram, fig, x=self._compute_unified_time_scale(spectrogram.shape[1]), axis=axs[1])
        #
        # # MFCC + d + a of the original speech signal
        # axs[2].set_title('Augmented MFCC + d + a #filters=13+13+13 of the original speech signal')
        # self._plot_pcolormesh(valid_originals, fig, x=self._compute_unified_time_scale(valid_originals.shape[1]), axis=axs[2])
        #
        # # Softmax of distances computed in VQ
        # axs[3].set_title('Softmax of distances computed in VQ\n($||z_e(x) - e_i||^2_2$ with $z_e(x)$ the output of the encoder prior to quantization)')
        # self._plot_pcolormesh(probs, fig, x=self._compute_unified_time_scale(probs.shape[1], downsampling_factor=2), axis=axs[3])
        #
        # encodings = evaluation_entry['encodings'].detach().cpu().numpy()
        # axs[4].set_title('Encodings')
        # self._plot_pcolormesh(encodings[0].transpose(), fig, x=self._compute_unified_time_scale(encodings[0].transpose().shape[1],
        #     downsampling_factor=2), axis=axs[4])

        # Actual reconstruction
        idx = [0.0625, 0.25, 1.0]
        for i in range(1, len(valid_reconstructions)+1):

            axs[i-1].set_title('Fraction of full code length:' + str(idx[i-1]))
            print("Reconstruction size:",valid_reconstructions[i-1].size())
            valid_reconstructions[i - 1] = valid_reconstructions[i - 1].detach().cpu().numpy()[0]
            d_1 = {' ': np.arange(len(valid_reconstructions[i-1])), '  ': valid_reconstructions[i-1]}
            pdnumsqr_1 = pd.DataFrame(d_1)
            sns.lineplot(x=' ', y='  ', data=pdnumsqr_1, ax=axs[i-1])
            write("reconstruction_"+str(i-1)+".wav", 16000, valid_reconstructions[i-1])

        output_path = '_evaluation-comparaison-plot_3.pdf'
        print("Output path:",output_path)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _plot_pcolormesh(self, data, fig, x=None, y=None, axis=None):
        axis = plt.gca() if axis is None else axis # default axis if None
        x = np.arange(data.shape[1]) if x is None else x # default x shape if None
        y = np.arange(data.shape[0]) if y is None else y # default y shape if None
        c = axis.pcolormesh(x, y, data)
        fig.colorbar(c, ax=axis)

    def _compute_unified_time_scale(self, shape, winstep=0.01, downsampling_factor=1):
        return np.arange(shape) * winstep * downsampling_factor

    def _plot_distances_histogram(self, evaluation_entry):
        encoding_distances = evaluation_entry['encoding_distances'][0].detach().cpu().numpy()
        embedding_distances = evaluation_entry['embedding_distances'].detach().cpu().numpy()
        frames_vs_embedding_distances = evaluation_entry['frames_vs_embedding_distances'].detach()[0].cpu().transpose(0, 1).numpy().ravel()

        if self._configuration['verbose']:
            ConsoleLogger.status('encoding_distances[0].size(): {}'.format(encoding_distances.shape))
            ConsoleLogger.status('embedding_distances.size(): {}'.format(embedding_distances.shape))
            ConsoleLogger.status('frames_vs_embedding_distances[0].shape: {}'.format(frames_vs_embedding_distances.shape))

        fig, axs = plt.subplots(3, 1, figsize=(30, 20), sharex=True)

        axs[0].set_title('\n'.join(wrap('Histogram of the distances between the'
            ' encodings vectors', 60)))
        sns.distplot(encoding_distances, hist=True, kde=False, ax=axs[0], norm_hist=True)

        axs[1].set_title('\n'.join(wrap('Histogram of the distances between the'
            ' embeddings vectors', 60)))
        sns.distplot(embedding_distances, hist=True, kde=False, ax=axs[1], norm_hist=True)

        axs[2].set_title(
            'Histogram of the distances computed in'
            ' VQ\n($||z_e(x) - e_i||^2_2$ with $z_e(x)$ the output of the encoder'
            ' prior to quantization)'
        )
        sns.distplot(frames_vs_embedding_distances, hist=True, kde=False, ax=axs[2], norm_hist=True)

        output_path = self._results_path + os.sep + self._experiment_name + '_distances-histogram-plot.png'
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def _test_denormalization(self, evaluation_entry):
        valid_originals = evaluation_entry['valid_originals'].detach().cpu()[0].numpy()
        valid_reconstructions = evaluation_entry['valid_reconstructions'].detach().cpu().numpy()
        normalizer = self._data_stream.normalizer

        denormalized_valid_originals = (normalizer['train_std'] * valid_originals.transpose() + normalizer['train_mean']).transpose()
        denormalized_valid_reconstructions = (normalizer['train_std'] * valid_reconstructions.transpose() + normalizer['train_mean']).transpose()

        # TODO: Remove the deltas and the accelerations, remove the zeros because it's the
        # energy, and compute the distance between the two

        fig, axs = plt.subplots(4, 1, figsize=(30, 20), sharex=True)

        # MFCC + d + a of the original speech signal
        axs[0].set_title('Augmented MFCC + d + a #filters=13+13+13 of the original speech signal')
        self._plot_pcolormesh(valid_originals, fig, x=self._compute_unified_time_scale(valid_originals.shape[1]), axis=axs[0])

        # Actual reconstruction
        axs[1].set_title('Actual reconstruction')
        self._plot_pcolormesh(valid_reconstructions, fig, x=self._compute_unified_time_scale(valid_reconstructions.shape[1]), axis=axs[1])

        # Denormalization of the original speech signal
        axs[2].set_title('Denormalized target')
        self._plot_pcolormesh(denormalized_valid_originals, fig, x=self._compute_unified_time_scale(denormalized_valid_originals.shape[1]), axis=axs[2])

        # Denormalization of the original speech signal
        axs[3].set_title('Denormalized reconstruction')
        self._plot_pcolormesh(denormalized_valid_reconstructions, fig, x=self._compute_unified_time_scale(denormalized_valid_reconstructions.shape[1]), axis=axs[3])

        output_path = self._results_path + os.sep + self._experiment_name + '_test-denormalization-plot.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _many_to_one_mapping(self):
        # TODO: fix it for batch size greater than one

        tokens_selections = list()
        val_speaker_ids = set()

        with tqdm(self._data_stream.validation_loader) as bar:
            for data in bar:
                valid_originals = data['input_features'].to(self._device).permute(0, 2, 1).contiguous().float()
                speaker_ids = data['speaker_id'].to(self._device)
                shifting_times = data['shifting_time'].to(self._device)
                wav_filenames = data['wav_filename']

                speaker_id = wav_filenames[0][0].split(os.sep)[-2]
                val_speaker_ids.add(speaker_id)

                if speaker_id not in os.listdir(self._vctk.raw_folder + os.sep + 'VCTK-Corpus' + os.sep + 'phonemes'):
                    # TODO: log the missing folders
                    continue

                z = self._model.encoder(valid_originals)
                z = self._model.pre_vq_conv(z)
                _, quantized, _, encodings, _, encoding_indices, _, \
                    _, _, _, _ = self._model.vq(z)
                valid_reconstructions = self._model.decoder(quantized, self._data_stream.speaker_dic, speaker_ids)
                B = valid_reconstructions.size(0)

                encoding_indices = encoding_indices.view(B, -1, 1)

                for i in range(len(valid_reconstructions)):
                    wav_filename = wav_filenames[0][i]
                    utterence_key = wav_filename.split('/')[-1].replace('.wav', '')
                    phonemes_alignment_path = os.sep.join(wav_filename.split('/')[:-3]) + os.sep + 'phonemes' + os.sep + utterence_key.split('_')[0] + os.sep \
                        + utterence_key + '.TextGrid'
                    tg = textgrid.TextGrid()
                    tg.read(phonemes_alignment_path)
                    entry = {
                        'encoding_indices': encoding_indices[i].detach().cpu().numpy(),
                        'groundtruth': tg.tiers[1],
                        'shifting_time': shifting_times[i].detach().cpu().item()
                    }
                    tokens_selections.append(entry)

        ConsoleLogger.status(val_speaker_ids)

        ConsoleLogger.status('{} tokens selections retreived'.format(len(tokens_selections)))

        phonemes_mapping = dict()
        # For each tokens selections (i.e. the number of valuations)
        for entry in tokens_selections:
            encoding_indices = entry['encoding_indices']
            unified_encoding_indices_time_scale = self._compute_unified_time_scale(
                encoding_indices.shape[0], downsampling_factor=2) # Compute the time scale array for each token
            """
            Search the grountruth phoneme where the selected token index time scale
            is within the groundtruth interval.
            Then, it adds the selected token index in the list of indices selected for
            the a specific token in the tokens mapping dictionnary.
            """
            for i in range(len(unified_encoding_indices_time_scale)):
                index_time_scale = unified_encoding_indices_time_scale[i] + entry['shifting_time']
                corresponding_phoneme = None
                for interval in entry['groundtruth']:
                    # TODO: replace that by nearest interpolation
                    if index_time_scale >= interval.minTime and index_time_scale <= interval.maxTime:
                        corresponding_phoneme = interval.mark
                        break
                if not corresponding_phoneme:
                    ConsoleLogger.warn("Corresponding phoneme not found. unified_encoding_indices_time_scale[{}]: {}"
                        "entry['shifting_time']: {} index_time_scale: {}".format(i, unified_encoding_indices_time_scale[i],
                        entry['shifting_time'], index_time_scale))
                if corresponding_phoneme not in phonemes_mapping:
                    phonemes_mapping[corresponding_phoneme] = list()
                phonemes_mapping[corresponding_phoneme].append(encoding_indices[i][0])

        ConsoleLogger.status('phonemes_mapping: {}'.format(phonemes_mapping))

        tokens_mapping = dict() # dictionnary that will contain the distribution for each token to fits with a certain phoneme

        """
        Fill the tokens_mapping such that for each token index (key)
        it contains the list of tuple of (phoneme, prob) where prob
        is the probability that the token fits this phoneme.
        """
        for phoneme, indices in phonemes_mapping.items():
            for index in list(set(indices)):
                if index not in tokens_mapping:
                    tokens_mapping[index] = list()
                tokens_mapping[index].append((phoneme, indices.count(index) / len(indices)))

        # Sort the probabilities for each token 
        for index, distribution in tokens_mapping.items():
            tokens_mapping[index] = list(sorted(distribution, key = lambda x: x[1], reverse=True))

        ConsoleLogger.status('tokens_mapping: {}'.format(tokens_mapping))

        with open(self._results_path + os.sep + self._experiment_name + '_phonemes_mapping.pickle', 'wb') as f:
            pickle.dump(phonemes_mapping, f)

        with open(self._results_path + os.sep + self._experiment_name + '_tokens_mapping.pickle', 'wb') as f:
            pickle.dump(tokens_mapping, f)

    def _compute_speaker_dependency_stats(self):
        """
        The goal of this function is to investiguate wether or not the supposed
        phonemes stored in the embeddings space are speaker independents.
        The algorithm is as follow:
            - Evaluate the model using the val dataset. Save each resulting
              embedding, with the corresponding speaker;
            - Group the embeddings by speaker;
            - Compute the distribution of each embedding;
            - Compute all the distances between all possible distribution couples, using
              a distribution distance (e.g. entropy) and plot them.
        """
        all_speaker_ids = list()
        all_embeddings = torch.tensor([]).to(self._device)

        with tqdm(self._data_stream.validation_loader) as bar:
            for data in bar:
                valid_originals = data['input_features'].to(self._device).permute(0, 2, 1).contiguous().float()
                speaker_ids = data['speaker_id'].to(self._device)
                wav_filenames = data['wav_filename']

                z = self._model.encoder(valid_originals)
                z = self._model.pre_vq_conv(z)
                _, quantized, _, _, _, _, _, \
                    _, _, _, _ = self._model.vq(z)
                valid_reconstructions = self._model.decoder(quantized, self._data_stream.speaker_dic, speaker_ids)
                B = valid_reconstructions.size(0)

                all_speaker_ids.append(speaker_ids.detach().cpu().numpy().tolist())
                #torch.cat(all_embeddings, self._model.vq.embedding.weight.data) # FIXME

        # - Group the embeddings by speaker: create a tensor/numpy per speaker id from all_embeddings
        # - Compute the distribution of each embedding (seaborn histogram, softmax)
        # - Compute all the distances between all possible distribution couples, using
        #   a distribution distance (e.g. entropy) and plot them (seaborn histogram?)

        # Snippet
        #_embedding_distances = [torch.dist(items[0], items[1], 2).to(self._device) for items in combinations(self._embedding.weight, r=2)]
        #embedding_distances = torch.tensor(_embedding_distances).to(self._device)

    def _compute_entropy_distributions(self):
        original_distribution = list()
        quantized_distribution = list()
        reconstruction_distribution = list()

        with tqdm(self._data_stream.validation_loader) as bar:
            for data in bar:
                valid_originals = data['input_features'].to(self._device).permute(0, 2, 1).contiguous().float()
                speaker_ids = data['speaker_id'].to(self._device)

                original_probs = F.softmax(valid_originals[0], dim=0).detach().cpu()
                original_entropy = -torch.sum(original_probs * torch.log(original_probs + 1e-10))

                z = self._model.encoder(valid_originals)
                z = self._model.pre_vq_conv(z)
                _, quantized, _, _, _, _, _, \
                    _, _, _, _ = self._model.vq(z)
                valid_reconstructions = self._model.decoder(quantized, self._data_stream.speaker_dic, speaker_ids)

                quantized_probs = F.softmax(quantized[0], dim=1).detach().cpu()
                quantized_entropy = -torch.sum(quantized_probs * torch.log(quantized_probs + 1e-10))

                reconstruction_probs = F.softmax(valid_reconstructions[0], dim=0).detach().cpu()
                reconstruction_entropy = -torch.sum(reconstruction_probs * torch.log(reconstruction_probs + 1e-10))

                original_distribution.append(original_entropy.detach().cpu().numpy())
                quantized_distribution.append(quantized_entropy.detach().cpu().numpy())
                reconstruction_distribution.append(reconstruction_entropy.detach().cpu().numpy())

        fig, axs = plt.subplots(3, 1, figsize=(30, 20), sharex=True)

        original_distribution = np.asarray(original_distribution).ravel()
        quantized_distribution = np.asarray(quantized_distribution).ravel()
        reconstruction_distribution = np.asarray(reconstruction_distribution).ravel()

        def dump_distribution(results_path, experiment_name, distribution_name, distribution):
            with open(results_path + os.sep + experiment_name + '_' + distribution_name + '.pickle', 'wb') as f:
                pickle.dump(distribution_name, f)

        dump_distribution(self._results_path, self._experiment_name, 'original_distribution', original_distribution)
        dump_distribution(self._results_path, self._experiment_name, 'quantized_distribution', quantized_distribution)
        dump_distribution(self._results_path, self._experiment_name, 'reconstruction_distribution', reconstruction_distribution)

        sns.distplot(original_distribution, hist=True, kde=False, ax=axs[0], norm_hist=True)
        axs[0].set_title('Entropy distribution of validation dataset')

        sns.distplot(quantized_distribution, hist=True, kde=False, ax=axs[1], norm_hist=True)
        axs[1].set_title('Entropy distribution of quantized validation dataset')

        sns.distplot(reconstruction_distribution, hist=True, kde=False, ax=axs[2], norm_hist=True)
        axs[2].set_title('Entropy distribution of reconstructed validation dataset')

        output_path = self._results_path + os.sep + self._experiment_name + '_entropy-stats-plot.png'
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
