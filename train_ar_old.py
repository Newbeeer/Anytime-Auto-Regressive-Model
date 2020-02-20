import torch
from modules import VectorQuantizedVAE_Dim, LSTM
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms, datasets
from datasets import MiniImagenet
from torchvision.utils import save_image, make_grid
from evaluation import calculate_fid_given_paths
from transfomer import TransformerModel
from warmup_scheduler import GradualWarmupScheduler
criterion = torch.nn.CrossEntropyLoss()


def count_parameters(model):
    # for p in model.parameters():
    #     print(p.size())
    # print("---------------------")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_samples_fetch(fix_images, model):
    x_tilde, z_e_x, z_q_x, indices = model(fix_images.cuda(), index=True)
    with torch.no_grad():
        x_tilde = model.indices_fetch(indices)
    return x_tilde


def generate_samples(epoch, prior, model, ar, args, writer):
    indices = []
    model.eval()
    ar.eval()
    for i in range(16):
        index = ar.sample(prior, args.hidden_size, torch.tensor([i % 10]).cuda())
        indices.append(index)

    indices = torch.tensor(indices).cuda()
    print("Sample indices:", indices)
    with torch.no_grad():
        idx_list = [20, 40, 60, 80, args.hidden_size]
        for i in idx_list:
            x_tilde = model.indices_fetch(indices[:, :i])
            grid = make_grid(x_tilde.cpu(), nrow=8, range=(-1, 1), normalize=True)
            writer.add_image('reconstruction_'+str(i), grid, epoch)
    return x_tilde


def cal_prior(data_loader, model):
    model.eval()
    print("Calculate Prior")
    prior = torch.zeros((args.k))
    for images, label in tqdm(data_loader):
        images = images.to(args.device)
        x_tilde, z_e_x, z_q_x, indices = model(images, index=True)
        for i in range(indices.size(0)):
            prior[int(indices[i][0])] += 1
    return prior / prior.sum()


def dump_codebook(data_loader, model):
    model.eval()
    print('Dump Codebook')
    lst = []
    for images, label in tqdm(data_loader):
        images = images.to(args.device)
        x_tilde, z_e_x, z_q_x, indices = model(images, index=True)
        lst.append(indices.cpu())
    return torch.cat(lst, 0)


def train(epoch, data_loader, model, ar,optimizer, args, writer, scheduler=None):
    model.eval()
    ar.train()

    for images, label in tqdm(data_loader):
        images = images.to(args.device)
        label = label.to(args.device)

        optimizer.zero_grad()
        with torch.no_grad():
            x_tilde, z_e_x, z_q_x, indices = model(images, index=True)
        # indices : (batch size , seq_len)
        # output : batch size * C, code size
        if args.model == 'lstm':
            output, _ = ar(indices, label)
            output = output.reshape(images.size(0), int(output.size(0) // images.size(0)), output.size(1))[:, :-1]
            output = output.contiguous().view(-1, output.size(2))
            indices = indices[:, 1:]
            indices = indices.contiguous().view(-1)

        elif args.model == 'transformer':
            # indices : B*T
            indices = indices.transpose(0,1)
            output = ar(indices)
            output = output.permute(1,0,2)[:, :-1]
            output = output.contiguous().view(-1, output.size(2))
            indices = indices.transpose(0,1)[:, 1:]
            indices = indices.contiguous().view(-1)

        loss = criterion(output, indices.long())
        loss.backward()

        # Logs
        writer.add_scalar('loss/train', loss.item(), args.steps)
        optimizer.step()
        if args.model == 'transformer':
            scheduler.step()
        args.steps += 1


def test(data_loader, model, ar, args, writer):
    loss = 0.
    model.eval()
    ar.eval()
    with torch.no_grad():
        for images, label in data_loader:
            images = images.to(args.device)
            label = label.to(args.device)
            x_tilde, z_e_x, z_q_x, indices = model(images, index=True)

            if args.model == 'lstm':
                output, _ = ar(indices, label)
                output = output.reshape(images.size(0), int(output.size(0) // images.size(0)), output.size(1))[:, :-1, :]
                output = output.contiguous().view(-1, output.size(2))
                indices = indices[:, 1:]
                indices = indices.contiguous().view(-1)

            elif args.model == 'transformer':
                indices = indices.transpose(0, 1)
                output = ar(indices)
                output = output.permute(1, 0, 2)[:, :-1]
                output = output.contiguous().view(-1, output.size(2))
                indices = indices.transpose(0, 1)[:, 1:]
                indices = indices.contiguous().view(-1)
            loss += criterion(output, indices.long())

    loss /= len(data_loader)
    # Logs
    writer.add_scalar('loss/test', loss.item(), args.steps)
    return loss


def main(args):

    writer = SummaryWriter(os.path.join(args.output_folder, 'log'))
    save_filename = os.path.join(args.output_folder, 'model')

    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if args.dataset == 'mnist':
            # Define the train & test datasets
            train_dataset = datasets.MNIST(args.data_folder, train=True,
                download=True, transform=transform)
            test_dataset = datasets.MNIST(args.data_folder, train=False, transform=transform)
            num_channels = 1
        elif args.dataset == 'fashion-mnist':
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            # Define the train & test datasets
            train_dataset = datasets.FashionMNIST(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(args.data_folder,
                train=False, transform=transform)
            num_channels = 1
        elif args.dataset == 'cifar10':

            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            # Define the train & test datasets
            train_dataset = datasets.CIFAR10(args.data_folder, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(args.data_folder, train=False, transform=transform)
            num_channels = 3
        valid_dataset = test_dataset
    elif args.dataset == 'miniimagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(args.data_folder, train=True, download=True, transform=transform)
        valid_dataset = MiniImagenet(args.data_folder, valid=True,download=True, transform=transform)
        test_dataset = MiniImagenet(args.data_folder, test=True, download=True, transform=transform)
        num_channels = 3

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=16, shuffle=True)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(train_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = VectorQuantizedVAE_Dim(num_channels, args.hidden_size, args.k).to(args.device)
    model.load_state_dict(torch.load(args.encoder_path))
    if args.model == 'lstm':
        regress_model = LSTM(args.k, args.embedding, args.lstm_h).to(args.device)
        optimizer = torch.optim.Adam(regress_model.parameters(), lr=args.lr, amsgrad=True)
        scheduler_warmup = None
    elif args.model == 'transformer':
        regress_model = TransformerModel(ntoken=args.k, ninp=512, nhead=8, nhid=2048, nlayers=6,dropout=0.1).cuda()
        optimizer = torch.optim.Adam(regress_model.parameters(), lr=args.lr, betas=(0.9, 0.999),eps=1e-6)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (50000 // 128) * args.num_epochs)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=(50000 // 128) * 6,after_scheduler=scheduler_cosine)

    # loss = test(valid_loader, model, regress_model, args, writer=None)
    # print("Loss:",loss)
    print("Trainable parameters for VAE:{}, for LSTM:{}".format(count_parameters(model),count_parameters(regress_model)))
    torch.save(dump_codebook(train_loader, model), os.path.join(save_filename, 'train.pt'))
    torch.save(dump_codebook(valid_loader, model), os.path.join(save_filename, 'valid.pt'))
    os._exit(0)
    prior = cal_prior(train_loader, model)
    # Generate the samples first once
    generate_samples(0, prior, model, regress_model, args, writer)
    best_loss = -1.
    for epoch in range(args.num_epochs):
        # if epoch >= 10:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(epoch, train_loader, model, regress_model, optimizer, args, writer, scheduler=scheduler_warmup)
        loss = test(valid_loader, model, regress_model, args, writer)
        if (epoch+1) % 5 == 0:
            fid = calculate_fid_given_paths(ar=regress_model, vqvae=model, dims=2048, prior=prior)
            print("Epoch:{}, Fid:{}".format(epoch, fid))
            writer.add_scalar('fid', fid, epoch)
        generate_samples(epoch + 1, prior, model, regress_model, args, writer)
        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(regress_model.state_dict(), f)
        with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
            torch.save(regress_model.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str, help='name of the data folder')
    parser.add_argument('--encoder-path', type=str, help='path to the encoder checkpoint')
    parser.add_argument('--dataset', type=str, help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')
    parser.add_argument('--model', type=str, help='name of the model (lstm, transformer)')

    # Latent space
    parser.add_argument('--hidden-size', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--embedding', type=int, default=1000,
                        help='size of the lstm embedding (default: 1000)')
    parser.add_argument('--lstm-h', type=int, default=500,
                        help='size of the lstm hidden (default: 500)')
    parser.add_argument('--k', type=int, default=500,
        help='number of latent vectors (default: 512)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=200,
        help='number of epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=2e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='vqvae',
        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cuda',
        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'log'), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, 'model'), exist_ok=True)
    args.steps = 0
    main(args)
