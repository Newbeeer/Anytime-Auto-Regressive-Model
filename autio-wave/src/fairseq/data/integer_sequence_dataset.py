import numpy as np
import torch

from . import FairseqDataset


def collate(samples):
    if len(samples) == 0:
        return {}

    src_tokens = torch.stack([s['source'] for s in samples])
    if samples[0]['target'] is None:
        target = None
    else:
        target = torch.stack([s['target'] for s in samples])

    return {
        'id': torch.LongTensor([s['id'] for s in samples]),
        'nsentences': len(samples),
        'ntokens': sum(len(s['source']) for s in samples),
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': torch.LongTensor([
                s['source'].numel() for s in samples
            ]),
        },
        'target': target,
    }


class IntegerSequenceDataset(FairseqDataset):

    def __init__(self, path_or_matrix, shuffle, bos_idx, source_only=False):
        if isinstance(path_or_matrix, str):
            path_or_matrix = torch.load(
                path_or_matrix, map_location=(lambda s, _: torch.serialization.default_restore_location(s, 'cpu')))
        self.matrix = path_or_matrix
        self.shuffle = shuffle
        self.bos_idx = bos_idx
        self.source_only = source_only

    def __getitem__(self, index):
        source = torch.cat([
            self.matrix.new_tensor([self.bos_idx]),
            self.matrix[index][:-1]
        ])
        if self.source_only:
            target = None
        else:
            target = self.matrix[index]
        return {'id': index, 'source': source, 'target': target}

    def __len__(self):
        return self.matrix.size(0)

    @staticmethod
    def get_empty_batch(bsz):
        return {
            'id': torch.arange(bsz),
            'nsentences': bsz,
            'ntokens': 0,
            'net_input': {
                'src_tokens': torch.empty((bsz, 0)).long(),
                'src_lengths': torch.zeros(bsz).long()
            },
            'target': torch.empty((bsz, 0)).long()
        }

    def collater(self, samples):
        return collate(samples)

    def num_tokens(self, index):
        return self.matrix.size(1)

    def size(self, index):
        return self.matrix.size(1)

    def ordered_indices(self):
        if self.shuffle:
            return np.random.permutation(len(self))
        else:
            return np.arange(len(self))

    @property
    def supports_prefetch(self):
        return False

    def prefetch(self, indices):
        raise NotImplementedError
