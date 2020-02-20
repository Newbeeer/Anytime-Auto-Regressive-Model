import torch
import copy
from fairseq.sequence_generator import SequenceGenerator

def sample(beam: int = 1, verbose: bool = False, **kwargs) -> str:
    # input = self.encode(sentence)

    # prior sample
    input = torch.tensor([4])
    hypo = generate(input, beam, verbose, **kwargs)[0]['tokens']
    return hypo

def generate(self, tokens: torch.LongTensor, beam: int = 5, verbose: bool = False, **kwargs) -> torch.LongTensor:
    sample = self._build_sample(tokens)

    # build generator using current args as well as any kwargs
    # gen_args = copy.copy(self.args)
    # gen_args.beam = beam
    # for k, v in kwargs.items():
    #     setattr(gen_args, k, v)
    # generator = self.task.build_generator(gen_args)

    translations = self.task.inference_step(generator, self.models, sample)

    if verbose:
        src_str_with_unk = self.string(tokens)
        print('S\t{}'.format(src_str_with_unk))

    # Process top predictions
    hypos = translations[0]

    return hypos