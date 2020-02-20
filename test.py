import argparse
from transformer_lm import TransformerLanguageModel
from fairseq import hub_utils
from fairseq.sequence_generator import SequenceGenerator
from fairseq.data.dictionary import Dictionary
import torch
sample_dict = {
        'id': torch.arange(10).long(),
        'nsentences': 10,
        'ntokens': 10,
        'net_input': {
            'src_tokens': torch.arange(10).long().unsqueeze(1),
            'src_lengths': torch.ones((10)).long(),
        },
        'target': 0,
    }
D = Dictionary(pad=-1,eos=-1,unk=-1,bos=-1)
parser2 = argparse.ArgumentParser(description='Transformer')
TransformerLanguageModel.add_args(parser2)
args = parser2.parse_args()
args.output_folder = 'transformer-test'
model_lm = TransformerLanguageModel.build_model(args, vocab_size=500)
G = SequenceGenerator(D)
G.generate([model_lm], sample_dict)
import torch
from modules import VectorQuantizedVAE_Dim, LSTM
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from transfomer import TransformerModel
from evaluation import calculate_fid_given_paths
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
        images = images.cuda()
        x_tilde, z_e_x, z_q_x, indices = model(images, index=True)
        for i in range(indices.size(0)):
            prior[int(indices[i][0])] += 1
    return prior / prior.sum()


def train(epoch, data_loader, model, ar,optimizer, args, writer, scheduler=None):
    model.eval()
    ar.train()

    for images, label in tqdm(data_loader):
        images = images.cuda()
        label = label.cuda()

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

            output, _ = ar(indices)
            print("Transformer output size:",output.size())
            output = output[:, :-1]
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
            images = images.cuda()
            label = label.cuda()
            x_tilde, z_e_x, z_q_x, indices = model(images, index=True)

            if args.model == 'lstm':
                output, _ = ar(indices, label)
                output = output.reshape(images.size(0), int(output.size(0) // images.size(0)), output.size(1))[:, :-1, :]
                output = output.contiguous().view(-1, output.size(2))
                indices = indices[:, 1:]
                indices = indices.contiguous().view(-1)

            elif args.model == 'transformer':
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

    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_filename = '/data1/xuyilun/models/{0}'.format(args.output_folder)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Define the train & test datasets
    train_dataset = datasets.CIFAR10('../cifar', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('../cifar', train=False, transform=transform)
    num_channels = 3
    valid_dataset = test_dataset

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=128, shuffle=False,
        num_workers=16, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=128, shuffle=False, drop_last=True,
        num_workers=16, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=16, shuffle=True)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(train_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = VectorQuantizedVAE_Dim(num_channels, args.hidden_size, args.k).cuda()
    model.load_state_dict(torch.load('/data1/xuyilun/models/cifar-h100k500--u-lr-3-ada4/best.pt'))
    if args.model == 'lstm':
        regress_model = LSTM(args.k, args.embedding, args.lstm_h).cuda()
        optimizer = torch.optim.Adam(regress_model.parameters(), lr=args.lr, amsgrad=True)
        scheduler_warmup = None
    elif args.model == 'transformer':
        regress_model = model_lm.cuda()
        optimizer = torch.optim.Adam(regress_model.parameters(), lr=args.lr, betas=(0.9, 0.999),eps=1e-6)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (50000 // 128) * args.num_epochs)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=(50000 // 128) * 6,after_scheduler=scheduler_cosine)

    print("Trainable parameters for VAE:{}, for LSTM:{}".format(count_parameters(model),count_parameters(regress_model)))
    prior = cal_prior(train_loader, model)
    # Generate the samples first once
    best_loss = -1
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

    args.steps = 0
    args.hidden_size = 100
    args.k = 500
    args.lr = 2e-4
    args.num_epochs = 100
    args.model = 'transformer'
    if not os.path.exists('/data1/xuyilun/models/{0}'.format(args.output_folder)):
        os.makedirs('/data1/xuyilun/models/{0}'.format(args.output_folder))
    args.steps = 0
    main(args)