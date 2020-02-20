import argparse
import os
import time

import imageio
import numpy as np
import torch
from torchvision.utils import make_grid
from fairseq import checkpoint_utils, utils
from modules import VectorQuantizedVAE_Dim, VectorQuantizedVAE_CelebA


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tokens-per-sample', type=int, required=True)
    parser.add_argument('--vocab-size', type=int, required=True)
    parser.add_argument('--n-samples', type=int, default=50000)
    parser.add_argument('--out-path', type=str, required=True)

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
    os.makedirs(args.out_path, exist_ok=True)
    ar_model = list()
    print('Loading AR model...')
    for i in [100]:
        ar_model.append(setup_ar(args.ar_checkpoint, args.ar_batch_size, i, args.ar_fp16))
    print('AR model loaded.')

    print('Loading AE model...')
    if args.ae_celeba:
        ae_model = VectorQuantizedVAE_CelebA(args.ae_input_channels, args.tokens_per_sample, args.vocab_size).cuda()
    else:
        ae_model = VectorQuantizedVAE_Dim(args.ae_input_channels, args.tokens_per_sample, args.vocab_size).cuda()
    ae_model.load_state_dict(torch.load(args.ae_checkpoint))
    ae_model.eval()
    print('AE model loaded.')
    if not os.path.exists(os.path.join('./generated_images', args.out_path)):
        os.mkdir(os.path.join('./generated_images', args.out_path))
    for i in range(10):
        with torch.no_grad():
            sampled_indices = torch.LongTensor(ar_model[0](16)).cuda()
        for j in [1]:
            with torch.no_grad():
                x_tilde = ae_model.indices_fetch(sampled_indices[:, :j])
            grid = make_grid(x_tilde, nrow=8, range=(-1, 1), normalize=True)
            grid = grid.permute(1, 2, 0)
            grid = ((grid) * 255).cpu().numpy().astype(np.uint8)
            imageio.imwrite(os.path.join('./generated_images', args.out_path, str(i) + '-' + str(j) + '.png'), grid)


if __name__ == '__main__':
    main(get_args())
