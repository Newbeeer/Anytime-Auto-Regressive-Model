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

from experiments.experiment import Experiment
from error_handling.console_logger import ConsoleLogger
from evaluation.alignment_stats import AlignmentStats
from evaluation.embedding_space_stats import EmbeddingSpaceStats
from evaluation.gradient_stats import GradientStats

import json
import yaml
import torch
import numpy as np
import random
import pickle
import os


class Experiments(object):

    def __init__(self, experiments):
        self._experiments = experiments

    @property
    def experiments(self):
        return self._experiments

    def train(self):
        for experiment in self._experiments:
            Experiments.set_deterministic_on(experiment.seed)
            experiment.train()
            torch.cuda.empty_cache()

    def eval(self):
        for experiment in self._experiments:
            Experiments.set_deterministic_on(experiment.seed)
            experiment.eval()
            torch.cuda.empty_cache()

    def dump(self):
        for experiment in self._experiments:
            Experiments.set_deterministic_on(experiment.seed)
            experiment.dump()
            torch.cuda.empty_cache()

    def fetch(self):
        for experiment in self._experiments:
            Experiments.set_deterministic_on(experiment.seed)
            experiment.fetch()
            torch.cuda.empty_cache()

    def evaluate(self, evaluation_options):
        # TODO: put all types of evaluation in evaluation_options, and skip this loop if none of them are set to true
        for experiment in self._experiments:
            Experiments.set_deterministic_on(experiment.seed)
            experiment.evaluate(evaluation_options)
            torch.cuda.empty_cache()

        if type(self._experiments[0].seed) == list:
            Experiments.set_deterministic_on(self._experiments[0].seed[0]) # For now use only the first seed there
        else:
            Experiments.set_deterministic_on(self._experiments[0].seed)

        if evaluation_options['compute_quantized_embedding_spaces_animation']:
            EmbeddingSpaceStats.compute_quantized_embedding_spaces_animation(
                all_experiments_paths=[experiment.experiment_path for experiment in self._experiments],
                all_experiments_names=[experiment.name for experiment in self._experiments],
                all_results_paths=[experiment.results_path for experiment in self._experiments]
            )

        if evaluation_options['plot_clustering_metrics_evolution']:
            AlignmentStats.compute_clustering_metrics_evolution(
                all_experiments_names=[experiment.name for experiment in self._experiments],
                result_path=self._experiments[0].results_path
            )

        if evaluation_options['check_clustering_metrics_stability_over_seeds']:
            AlignmentStats.check_clustering_metrics_stability_over_seeds(
                all_experiments_names=[experiment.name for experiment in self._experiments],
                result_path=self._experiments[0].results_path
            )

        if evaluation_options['plot_gradient_stats']:
            all_experiments_paths = [experiment.experiment_path for experiment in self._experiments]
            all_experiments_names = [experiment.name for experiment in self._experiments]
            all_results_paths = [experiment.results_path for experiment in self._experiments]
            gradient_stats_entries = list()
            for i in range(len(all_experiments_paths)):
                experiment_path = all_experiments_paths[i]
                experiment_name = all_experiments_names[i]
                experiment_results_path = all_results_paths[i]
                # List all file names related to the gradient stats for the current observed experiment
                file_names = [file_name for file_name in os.listdir(experiment_path) if 'gradient-stats' in file_name and experiment_name in file_name]

                # Sort file names by epoch number and iteration number as well
                file_names = sorted(file_names, key=lambda x: 
                    (int(x.replace(experiment_name + '_', '').replace('_gradient-stats.pickle', '').split('_')[0]),
                    int(x.replace(experiment_name + '_', '').replace('_gradient-stats.pickle', '').split('_')[1]))
                )

                with tqdm(file_names) as bar:
                    bar.set_description('Processing')
                    for file_name in bar:
                        with open(experiment_path + os.sep + file_name, 'rb') as file:
                            split_file_name = file_name.replace(experiment_name + '_', '').replace('_gradients-stats.pickle', '').split('_')
                            gradient_stats_entries.append((int(split_file_name[0]), int(split_file_name[1]), pickle.load(file)))

                GradientStats.plot_gradient_flow_over_epochs(
                    gradient_stats_entries,
                    output_file_name=experiment_results_path + os.sep + experiment_name + '_gradient_flow.png'
                )

    @staticmethod
    def set_deterministic_on(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def load(experiments_path):
        experiments = list()
        with open(experiments_path, 'r') as experiments_file:
            experiment_configurations = json.load(experiments_file)

            configuration = None
            with open(experiment_configurations['configuration_path'], 'r') as configuration_file:
                configuration = yaml.load(configuration_file, Loader=yaml.FullLoader)

            if type(experiment_configurations['seed']) == list:
                for seed in experiment_configurations['seed']:
                    for experiment_configuration_key in experiment_configurations['experiments'].keys():
                        experiment = Experiment(
                            name=experiment_configuration_key + '-seed' + str(seed),
                            experiments_path=experiment_configurations['experiments_path'],
                            results_path=experiment_configurations['results_path'],
                            global_configuration=configuration,
                            experiment_configuration=experiment_configurations['experiments'][experiment_configuration_key],
                            seed=seed
                        )
                        experiments.append(experiment)
            else:
                for experiment_configuration_key in experiment_configurations['experiments'].keys():
                    experiment = Experiment(
                        name=experiment_configuration_key,
                        experiments_path=experiment_configurations['experiments_path'],
                        results_path=experiment_configurations['results_path'],
                        global_configuration=configuration,
                        experiment_configuration=experiment_configurations['experiments'][experiment_configuration_key],
                        seed=experiment_configurations['seed']
                    )
                    experiments.append(experiment)

        return Experiments(experiments)
