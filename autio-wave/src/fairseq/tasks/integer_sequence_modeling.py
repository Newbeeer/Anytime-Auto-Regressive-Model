import os

import torch

from fairseq.data import DummyDictionary, IntegerSequenceDataset
from fairseq.tasks import FairseqTask, register_task


@register_task("integer_sequence_modeling")
class IntegerSequenceModelingTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--vocab-size', type=int, help='size of the vocabulary')
        parser.add_argument('--tokens-per-sample', default=1024, type=int,
                            help='max number of tokens per sample for LM dataset')
        # fmt: on

    def __init__(self, args):
        super().__init__(args)
        self.dictionary = DummyDictionary(self.args.vocab_size)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def build_model(self, args):
        model = super().build_model(args)
        return model

    def load_dataset(self, split, combine=False, **kwargs):
        split_path = os.path.join(self.args.data, split + '.pt')
        dataset = IntegerSequenceDataset(split_path, shuffle=True, bos_idx=self.dictionary.bos_index)
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )
        self.datasets[split] = dataset

    def build_dataset_for_inference(self, src_matrix):
        return IntegerSequenceDataset(
            src_matrix,
            shuffle=False,
            bos_idx=self.dictionary.bos_index,
            source_only=True
        )

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                if prefix_tokens[:, 0].eq(self.dictionary.bos_index).all():
                    prefix_tokens = prefix_tokens[:, 1:]
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
