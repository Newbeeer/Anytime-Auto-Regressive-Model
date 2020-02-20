import argparse
import torch
import sys
import time
import os
import logging
import yaml
import shutil
import numpy as np
import tensorboardX
import torch.optim as optim
import torchvision
from models import ImageTransformer
import matplotlib
import itertools
from utils import parse_args_and_config
from tqdm import tqdm
import torch.nn as nn

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

def parse_args():
    """
    : Create all flags for argument parser
    """
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, default='transformer_dmol.yml', help='Path to the config file')
    parser.add_argument('--doc', type=str, default='0', help='A string for documentation purpose')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--train_sample', action='store_true', help='Sample at train time')
    parser.add_argument('--infer', action='store_true', help='Reload a model specified by doc and generate samples')
    parser.add_argument('--resume', action='store_true', help='Reload a model specified by doc and resume training')
    parser.add_argument('--overfit', action='store_true', help='Overfit to a single batch for debug')
    args = parser.parse_args()
    return args

def get_lr(step, config):
    warmup_steps = config.optim.warmup
    lr_base = config.optim.lr * 0.002 # for Adam correction
    ret = 5000. * config.model.hidden_size ** (-0.5) * \
          np.min([(step + 1) * warmup_steps ** (-1.5), (step + 1) ** (-0.5)])
    return ret * lr_base

def total_grad_norm(params):
    total_norm = 0
    for p in params:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = (total_norm ** (1. / 2))
    return total_norm

class VarDequantDense(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.dense = nn.Linear(size, size)

    def forward(self, X):
        eps = torch.randn_like(X)
        y = self.dense(eps)
        s, jac = torch.slogdet(self.dense.weight)
        y = torch.sigmoid(y)
        jac = torch.log(y).sum(1) + torch.log(1. - y).sum(1) + jac
        log_pdf_eps = -torch.tensor(X.shape[1] * 0.5 * np.log(2 * np.pi), device=X.device) - 0.5 * torch.sum(eps ** 2, dim=1)
        return y, -log_pdf_eps + jac, eps

def delete_and_create_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

# TODO: Can also plot the marginals after reverting!
def eval_model(model, var_dequant, loader, converter, config, args, reverter, tb_logger, step):
    input_dim = config.model.image_size ** 2 * config.model.channels
    model.eval()
    all_pred_samples = []
    all_imgs = []
    all_noise = []
    all_converted_noise = []

    with torch.no_grad():
        for i, (imgs, _) in tqdm(enumerate(loader)):
            imgs = imgs.to(config.device)
            orig_imgs = torch.cat([imgs * 255. / 256 for _ in range(5)], 0)
            if config.model.distr in ["mol", "mog"]:
                if config.data.var_dequant:
                    dqed = []
                    for _ in range(5):
                        dqed_imgs, _ = var_dequant(imgs, torch.zeros(imgs.shape[0]).to(config.device))
                        dqed.append(dqed_imgs)
                else:
                    dqed = [(imgs * 255. + torch.rand_like(imgs)) / 256. for _ in range(5)]
                imgs = torch.cat(dqed, 0)
                # imgs = (imgs * 255. + torch.rand_like(imgs)) / 256. # Dequantization

            noise = imgs - orig_imgs
            orig_imgs, _ = converter(orig_imgs)
            imgs, _ = converter(imgs)
            all_imgs.append(imgs)
            converted_noise = imgs - orig_imgs
            all_noise.append(noise)
            all_converted_noise.append(converted_noise)

            preds = model(imgs)
            pred_samples = model.sample_from_preds(preds)
            all_pred_samples.append(pred_samples)

    all_pred_samples = torch.cat(all_pred_samples, 0)
    all_imgs = torch.cat(all_imgs, 0)
    noise = torch.cat(all_noise, 0).flatten(1)
    converted_noise = torch.cat(all_converted_noise, 0)

    if config.overfit:
        with torch.no_grad():
            samples = model.sample(25000, config.device)
        assert all_imgs.shape[0] % samples.shape[0] == 0
        delete_and_create_dir(config.log + "/marginals/{}".format(step))
        minimum = torch.min(all_imgs.min(0)[0], samples.min(0)[0])
        maximum = torch.max(all_imgs.max(0)[0], samples.max(0)[0])
        for dim in tqdm(range(input_dim)):
            plt.figure()
            plt.hist(samples[:, dim].cpu().numpy().repeat(all_imgs.shape[0] // samples.shape[0]),
                     alpha=0.5, label="model", bins=100, color='r',
                     range=(minimum[dim].item(), maximum[dim].item()))
            plt.hist(all_imgs[:, dim].cpu().numpy(), alpha=0.5, label="imgs", bins=100, color='b',
                     range=(minimum[dim].item(), maximum[dim].item()))
            plt.legend()
            plt.title("marginals of dim {}".format(dim))
            plt.savefig(config.log + "/marginals/{}/dim_{:03d}.jpg".format(step, dim))
            plt.close()

    delete_and_create_dir(config.log + "/combined/{}".format(step))
    minimum = torch.min(all_imgs.min(0)[0], all_pred_samples.min(0)[0])
    maximum = torch.max(all_imgs.max(0)[0], all_pred_samples.max(0)[0])
    for dim in tqdm(range(input_dim)):
        plt.figure()
        plt.hist(all_pred_samples[:, dim].cpu().numpy(), alpha=0.5, label="preds", bins=100, color='r',
                 range=(minimum[dim].item(), maximum[dim].item()))
        plt.hist(all_imgs[:, dim].cpu().numpy(), alpha=0.5, label="imgs", bins=100, color='b',
                 range=(minimum[dim].item(), maximum[dim].item()))
        plt.legend()
        plt.title("dim {}".format(dim))
        plt.savefig(config.log + "/combined/{}/dim_{:03d}.jpg".format(step, dim))
        plt.close()

    delete_and_create_dir(config.log + "/converted_noise/{}".format(step))
    for dim in tqdm(range(input_dim)):
        plt.figure()
        plt.hist(converted_noise[:, dim].cpu().numpy(), alpha=0.5, bins=100, color='r')
        plt.title("dim {}".format(dim))
        plt.savefig(config.log + "/converted_noise/{}/dim_{:03d}.jpg".format(step, dim))
        plt.close()

    delete_and_create_dir(config.log + "/noise/{}".format(step))
    for dim in tqdm(range(input_dim)):
        plt.figure()
        plt.hist(noise[:, dim].cpu().numpy(), alpha=0.5, bins=100, color='r')
        plt.title("dim {}".format(dim))
        plt.savefig(config.log + "/noise/{}/dim_{:03d}.jpg".format(step, dim))
        plt.close()

    # delete_and_create_dir(config.log + "/pred_hists")
    # for dim in tqdm(range(input_dim)):
    #     plt.figure()
    #     plt.hist(all_pred_samples[:, dim].cpu().numpy(), bins=200, range=(minimum[dim].item(), maximum[dim].item()))
    #     plt.title("dim {}".format(dim))
    #     plt.savefig(config.log + "/pred_hists/dim_{:03d}.jpg".format(dim))
    #     plt.close()
    #
    # delete_and_create_dir(config.log + "/hists")
    # for dim in tqdm(range(input_dim)):
    #     plt.figure()
    #     plt.hist(all_imgs[:, dim].cpu().numpy(), bins=200, range=(minimum[dim].item(), maximum[dim].item()))
    #     plt.title("dim {}".format(dim))
    #     plt.savefig(config.log + "/hists/dim_{:03d}.jpg".format(dim))
    #     plt.close()


def overfit_marginals_test(loader, var_dequant, tb_logger, config, converter, reverter, step):
    with torch.no_grad():
        input_dim = config.model.image_size ** 2 * config.model.channels
        all_imgs = []
        for _, (imgs, _) in enumerate(loader):
            imgs = imgs.to(config.device)
            if config.data.var_dequant:
                imgs, _ = var_dequant(imgs, torch.zeros(imgs.shape[0]).to(config.device))
            else:
                imgs = (imgs * 255. + torch.rand_like(imgs)) / 256.  # Dequantization
                _ = 0

            imgs, jac = converter(imgs)
            all_imgs.append(imgs)

        imgs = torch.cat(all_imgs, 0)

    corr = np.corrcoef(imgs.cpu().numpy().T)
    corr = np.abs(corr)
    corr_no_diag_mean = np.mean(corr - np.diag(np.diag(corr))) * input_dim / (input_dim - 1)
    logging.info("Avg corr with diag: {:0.3f}, Avg corr without diag: {:0.3f}".format(np.mean(corr), corr_no_diag_mean))
    tb_logger.add_scalar('corr', np.mean(corr), global_step=step)
    tb_logger.add_scalar('corr_no_diag', corr_no_diag_mean, global_step=step)

    for col in range(input_dim):
        idxs = torch.randperm(imgs.shape[0])
        imgs[:, col] = imgs[idxs, col]

    corr = np.corrcoef(imgs.cpu().numpy().T)
    corr = np.abs(corr)
    corr_no_diag_mean = np.mean(corr - np.diag(np.diag(corr))) * input_dim / (input_dim - 1)
    logging.info("Applied permutations to only preserve marginals")
    logging.info("Avg corr with diag: {:0.3f}, Avg corr without diag: {:0.3f}".format(np.mean(corr), corr_no_diag_mean))
    tb_logger.add_scalar('marg_corr', np.mean(corr), global_step=step)
    tb_logger.add_scalar('marg_corr_no_diag', corr_no_diag_mean, global_step=step)

    rev_margs = reverter(imgs[:25])
    imgs_grid = torchvision.utils.make_grid(torch.clamp(rev_margs[:25, ...], 0., 1.), 5)
    tb_logger.add_image('reverted_prod_marginals', imgs_grid, global_step=step)

def train(model,loader,config, args, tb_logger,  step):
    input_dim = config.model.image_size ** 2 * config.model.channels

    optimizer = optim.Adam(model.parameters(), lr=1., betas=(0.9, 0.98), eps=1e-9, amsgrad=config.optim.amsgrad)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step, config))
    losses_per_dim = torch.zeros(config.model.channels, config.model.image_size, config.model.image_size).to(config.device)

    for _ in range(config.train.epochs):
        for _, (imgs, _) in enumerate(loader):

            imgs = imgs.to(config.device)
            imgs = converter(imgs)

            model.train()
            scheduler.step()
            optimizer.zero_grad()

            preds = model(imgs)
            loss = model.loss(preds, imgs)
            decay = 0. if step == 0 else 0.9
            # NOTE: this doesn't mean too much now because of the logit transform -- should be adding appropriate
            # jac term to each pixels bpd.
            if config.model.distr == "dmol":
                losses_per_dim[0,:,:] = losses_per_dim[0,:,:] * decay + (1 - decay) * loss.detach().mean(0) / np.log(2)
            else:
                losses_per_dim = losses_per_dim * decay + (1 - decay) * loss.detach().mean(0).view(losses_per_dim.shape) / np.log(2)
            loss = loss.view(loss.shape[0], -1).sum(1)
            loss = loss.mean(0)
            loss.backward()

            total_norm = total_grad_norm(model.parameters())
            if config.train.clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)
            total_norm_post = total_grad_norm(model.parameters())
            optimizer.step()

            bits_per_dim = loss / (np.log(2.) * input_dim)
            acc = model.accuracy(preds, imgs)

            if step % config.train.log_iter == 0:
                logging.info('step: {}; loss: {:.3f}; bits_per_dim: {:.3f}, acc: {:.3f}, grad norm pre: {:.3f}, post: {:.3f}'
                             .format(step, loss.item(), bits_per_dim.item(), acc.item(), total_norm, total_norm_post))
                tb_logger.add_scalar('loss', loss.item(), global_step=step)
                tb_logger.add_scalar('bits_per_dim', bits_per_dim.item(), global_step=step)
                tb_logger.add_scalar('acc', acc.item(), global_step=step)
                tb_logger.add_scalar('grad_norm', total_norm, global_step=step)



            model.eval()
            step += 1


def sample(model, imgs, reverter, tb_logger, config, step):
    with torch.no_grad():
        samples = model.sample(config.train.sample_size, config.device)
        reverted_samples = reverter(samples)
        samples_grid = torchvision.utils.make_grid(torch.clamp(reverted_samples[:8, ...], 0., 1.), 3)
        tb_logger.add_image('samples', samples_grid, global_step=step)
        reverted_imgs = reverter(imgs)
        imgs_grid = torchvision.utils.make_grid(torch.clamp(reverted_imgs[:8, ...], 0., 1.), 3)
        tb_logger.add_image('imgs', imgs_grid, global_step=step)
        if config.overfit:
            l2_err = (reverted_samples - reverted_imgs).flatten(1).norm(dim=1).mean(0)
            logging.info("Sample l2 err (reverted): {}".format(l2_err.item()))
            tb_logger.add_scalar('reverted_sample_l2_err', l2_err.item(), global_step=step)
            # This is only interesting in the case of a flow model
            l2_err = (samples - imgs).flatten(1).norm(dim=1).mean(0)
            logging.info("Sample l2 err (non-reverted): {}".format(l2_err.item()))
            tb_logger.add_scalar('sample_l2_err', l2_err.item(), global_step=step)

        # manual_ranks = [10, 64, 128, 1024]
        manual_ranks = [2, 10, 32, 64, 100, 300]
        for rank in manual_ranks:
            lowrank_samples = samples.clone().view(samples.shape[0], -1)
            lowrank_samples[:,rank:] = 0.
            lowrank_samples = reverter(lowrank_samples.view(samples.shape))
            samples_grid = torchvision.utils.make_grid(torch.clamp(lowrank_samples[:8, ...], 0., 1.), 3)
            tb_logger.add_image('lowrank_samples/{}'.format(rank), samples_grid, global_step=step)

            lowrank_data = imgs.clone().view(imgs.shape[0], -1)
            lowrank_data[:,rank:] = 0.
            lowrank_data = reverter(lowrank_data.view(imgs.shape))
            data_grid = torchvision.utils.make_grid(torch.clamp(lowrank_data[:8, ...], 0., 1.), 3)
            tb_logger.add_image('lowrank_data/{}'.format(rank), data_grid, global_step=step)


def debug_data_stats(config, loader):
    # Accumulate data statistics for debugging purposes, e.g. to analyze the entropy of the first dimension
    data_avgs = torch.zeros(config.model.channels, config.model.image_size, config.model.image_size, 256)
    for i, (imgs, l) in tqdm(enumerate(loader)):
        one_hot_data = torch.zeros(imgs.shape + (256,)).scatter_(-1, (imgs * 255).long().unsqueeze(-1), 1)
        data_avgs += one_hot_data.mean(0)
    data_avgs /= i
    return data_avgs


def main():
    args = parse_args()
    args, config = parse_args_and_config(args, config_dir="../configs/transformer", logdir="../logs/transformer",
                                         resume=(args.resume or args.infer))

    tb_logger = tensorboardX.SummaryWriter(log_dir=os.path.join('../logs/transformer', args.doc))

    model = ImageTransformer(config.model).to(config.device)
    var_dequant = None

    if args.resume or args.infer:
        # TODO: when dequantizing the PCA matrix, I should be saving that as well for reloading.
        logging.info("Loading model from saved model")
        model.load_state_dict(torch.load(os.path.join(args.log, "model.pth")))
        if config.data.var_dequant:
            var_dequant.load_state_dict(torch.load(os.path.join(args.log, "var_dequant.pth")))

        # Initialize as in their code
        gain = config.model.initializer_gain
        for name, p in model.named_parameters():
            if "layernorm" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=np.sqrt(gain)) # Need sqrt for inconsistency between pytorch / TF
            else:
                a = np.sqrt(3. * gain / p.shape[0])
                nn.init.uniform_(p, -a, a)
        step = 0

    if not args.infer:
        train(model, train_loader, config, args,  tb_logger)

    eval_model(model,  train_loader, config, args,  tb_logger)
    imgs, _ = next(iter(train_loader))
    sample(model, imgs,  tb_logger, config, step=step)
    return 0

if __name__ == '__main__':
    sys.exit(main())