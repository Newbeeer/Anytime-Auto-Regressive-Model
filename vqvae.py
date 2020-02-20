import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

import datasets
from modules import VectorQuantizedVAE_Dim,VectorQuantizedVAE_CelebA,VectorQuantizedVAE_mnist


def train(epoch, data_loader, model, optimizer, args, writer):
    for images, _ in tqdm(data_loader):
        images = images.to(args.device)

        if args.pretrain:
            optimizer.zero_grad()
            x_tilde, z_e_x, z_q_x = model(images, full_sample=True)
            # Reconstruction loss
            loss_recons = F.mse_loss(x_tilde, images)
            # Vector quantization objective
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            # Commitment objective
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
            loss = loss_recons + loss_vq + loss_commit
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            x_tilde, z_e_x, z_q_x, index = model(images, full_sample=False)
            # Reconstruction loss
            loss_recons = F.mse_loss(x_tilde, images)
            # Vector quantization objective
            loss_vq = F.mse_loss(z_q_x[:, :index], z_e_x.detach()[:, :index])
            # Commitment objective
            loss_commit = F.mse_loss(z_e_x[:, :index], z_q_x.detach()[:, :index])
            loss = loss_recons + loss_vq + loss_commit
            loss.backward()
            optimizer.step()

        # Logs
        writer.add_scalar('loss/train/reconstruction',
                          loss_recons.item(), args.steps)
        writer.add_scalar('loss/train/quantization',
                          loss_vq.item(), args.steps)
        args.steps += 1


def test(data_loader, model, args, writer):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        loss_recons_designate = {
            args.hidden_size // 5 * (i + 1): 0
            for i in range(5 - 1)
        }
        for images, _ in data_loader:
            images = images.to(args.device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)
            for i in loss_recons_designate.keys():
                x_tilde, z_e_x, z_q_x, _ = model(images, full_sample=False, designate=i)
                loss_recons_designate[i] += F.mse_loss(x_tilde, images)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)
        for i in loss_recons_designate.keys():
            loss_recons_designate[i] /= len(data_loader)
            writer.add_scalar('loss/test/reconstruction_' + str(i), loss_recons_designate[i].item(), args.steps)
    # Logs
    writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
    writer.add_scalar('loss/test/quantization', loss_vq.item(), args.steps)

    return loss_recons.item(), loss_vq.item()


def generate_samples(images, model, args, full, designate):
    with torch.no_grad():
        images = images.to(args.device)
        if not full:
            x_tilde, _, _, _ = model(images, full, designate)
        else:
            x_tilde, _, _ = model(images, full, designate)
    return x_tilde


def generate_samples_train(data_loader, model, args, full, designate=None):
    print("Current designate:", designate)
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(data_loader):
            images = images.to(args.device)
            if not full:
                x_tilde, _, _, _ = model(images, full, designate)
            else:
                x_tilde, _, _ = model(images, full, designate)

            loss += F.mse_loss(x_tilde, images, reduction='sum')

    loss /= len(data_loader.dataset)
    return loss


def plot_reconstructed_images(fixed_images, model, epoch, writer):
    for i in range(5 - 1):
        hidden_step = args.hidden_size // 5 * (i + 1)
        reconstruction = generate_samples(fixed_images, model, args, False, hidden_step)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        writer.add_image(f'reconstruction_{hidden_step}', grid, epoch)

    reconstruction = generate_samples(fixed_images, model, args, True, None)
    grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('reconstruction_full', grid, epoch)


def main(args):
    writer = SummaryWriter('./logs/{0}'.format(args.out_path))
    save_filename = os.path.join('/data1/xuyilun', args.out_path, 'model')

    train_dataset, valid_dataset, test_dataset, num_channels = datasets.load_data(args.dataset, args.data_folder)

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)
    if args.dataset == 'celeba':
        model = VectorQuantizedVAE_CelebA(num_channels, args.hidden_size, args.k).to(args.device)
    elif args.dataaset == 'mnist':
        model = VectorQuantizedVAE_mnist(num_channels, args.hidden_size, args.k).to(args.device)
    else:
        model = VectorQuantizedVAE_Dim(num_channels, args.hidden_size, args.k).to(args.device)

    if args.restore_checkpoint:
        assert not args.pretrain
        model.load_state_dict(torch.load(args.restore_checkpoint))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, args.adam_beta2),
                                 eps=args.adam_eps, amsgrad=args.amsgrad)

    plot_reconstructed_images(fixed_images, model, 0, writer)

    best_loss = -1.
    loss, _ = test(valid_loader, model, args, writer)
    for epoch in range(args.num_epochs):
        train(epoch, train_loader, model, optimizer, args, writer)
        loss, _ = test(valid_loader, model, args, writer)

        plot_reconstructed_images(fixed_images, model, epoch + 1, writer)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f)
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(model.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str,
                        help='name of the data folder')
    parser.add_argument('--dataset', type=str,
                        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet, celeba)')
    parser.add_argument('--out-path', type=str,
                        help='output path')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=100,
                        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=500,
                        help='number of latent vectors (default: 500)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='learning rate for Adam optimizer (default: 2e-3)')
    parser.add_argument('--adam-beta2', type=float, default=0.999,
                        help='beta2 for Adam optimizer')
    parser.add_argument('--adam-eps', type=float, default=1e-8,
                        help='eps for Adam optimizer')
    parser.add_argument('--amsgrad', action='store_true',
                        help='turn on amsgrad for the optimizer')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cuda',
                        help='set the device (cpu or cuda, default: cpu)')
    parser.add_argument('--pretrain', default=False, action='store_true')
    parser.add_argument('--restore-checkpoint', type=str,
                        help='load pre-trained checkpoint')
    args = parser.parse_args()

    log_path = './logs'
    model_path = os.path.join('/data1/xuyilun', args.out_path, 'model')
    # Create logs and models folder if they don't exist
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    # Device
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Slurm
    args.steps = 0
    args.interval = torch.tensor(args.hidden_size)
    main(args)
