import argparse
import os
import time

import numpy as np
import torch

from fairseq import checkpoint_utils, utils


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tokens-per-sample', type=int, required=True)
    parser.add_argument('--tokens-per-target', type=int, default=None)
    parser.add_argument('--vocab-size', type=int, required=True)
    parser.add_argument('--n-samples', type=int, default=100000)
    parser.add_argument('--out-path', type=str, required=True)
    parser.add_argument('--original', action='store_true')

    parser.add_argument('--ae-checkpoint', type=str, required=True)
    parser.add_argument('--ae-batch-size', type=int, default=512)
    parser.add_argument('--ae-input-channels', type=int, default=3)
    parser.add_argument('--ae-celeba', action='store_true', default=False)

    parser.add_argument('--ar-checkpoint', type=str, required=True)
    parser.add_argument('--ar-batch-size', type=int, default=512)
    parser.add_argument('--ar-fp16', action='store_true')


    return parser.parse_args()


def build_sample(task, matrix=None, bs=None):
    if matrix is None:
        matrix = torch.empty((0, 1)).long()
    dataset = task.build_dataset_for_inference(matrix)
    if len(dataset) == 0:
        sample = dataset.get_empty_batch(bs)
    else:
        assert bs is None or bs == len(dataset)
        sample = dataset.collater(list(dataset))
    sample = utils.apply_to_sample(
        lambda tensor: tensor.cuda(),
        sample
    )
    return sample


def setup_ar(checkpoint, batch_size, tokens_per_target=None, fp16=False):
    models, args, task = checkpoint_utils.load_model_ensemble_and_task([checkpoint])

    if tokens_per_target is None:
        tokens_per_target = args.tokens_per_sample

    args_overrides = {
        'cpu': False,
        'no_beamable_mm': False,
        'max_sentences': batch_size,
        'fp16': fp16,
        'tokens_per_target': tokens_per_target,
        'sampling': True,
        'beam': 1
    }
    for arg_name, arg_val in args_overrides.items():
        setattr(args, arg_name, arg_val)
    use_cuda = torch.cuda.is_available() and not args.cpu
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.max_sentences,
            need_attn=False,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()
    generator = task.build_generator(args)

    def generate_fn(n):
        bs = task.args.max_sentences
        results = []
        for i in range(0, n, bs):
            real_bs = min(bs, n - i)
            sample = build_sample(task, bs=real_bs)
            tokens, _ = task.inference_step(generator, models, sample)
            results.append(tokens)
        return np.concatenate(results, axis=0)

    return generate_fn


def main(args):
    print('Loading AR model...')
    print("Sampling number:{}".format(args.tokens_per_target))
    print("AR batch size:",args.ar_batch_size)
    ar_generate_fn = setup_ar(args.ar_checkpoint, args.ar_batch_size, args.tokens_per_target, args.ar_fp16)
    print('AR model loaded.')


    print('Running AR...')
    start_time = time.time()
    sampled_indices = ar_generate_fn(args.n_samples)
    mid_time = time.time()
    print(f'AR finished in {mid_time - start_time} seconds.')
    np.save('indices.npy',sampled_indices)
    print("Done!")

if __name__ == '__main__':
    main(get_args())
