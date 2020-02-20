import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import numpy as np
from functions import vq, vq_st, vq_st_i, vq_st_ori, vq_ori, vq_st_i_ori

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VQEmbedding_Original(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.D = D
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0,2,3,1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x, index=False):

        z_e_x_ = z_e_x.permute(0,2,3,1).contiguous()

        if index:
            z_q_x_, indices, indices_not_flatten = vq_st_i_ori(z_e_x_, self.embedding.weight.detach())
        else:
            z_q_x_, indices = vq_st_ori(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0,3,1,2).contiguous()
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0,3,1,2).contiguous()
        if index:
            return z_q_x, z_q_x_bar, indices_not_flatten
        else:
            return z_q_x, z_q_x_bar

    def indices_fetch(self, indices):
        indices_flatten = indices.reshape(indices.size(0) * indices.size(1) * indices.size(2))
        z_q_x_fetch = torch.index_select(self.embedding.weight, dim=0, index=indices_flatten).view(indices.size(0), self.D, indices.size(1), indices.size(1))  # B*C*H*W

        return z_q_x_fetch


class VQEmbedding(nn.Module):
    def __init__(self, K, H, W):
        super().__init__()
        self.H = H
        self.embedding = nn.Embedding(K, H * W)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x, index=False):

        z_e_x_ = z_e_x.contiguous()

        if index:
            z_q_x_, indices, indices_not_flatten = vq_st_i(z_e_x_, self.embedding.weight.detach())
        else:
            z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())

        z_q_x = z_q_x_.contiguous()
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.contiguous()
        if index:
            return z_q_x, z_q_x_bar, indices_not_flatten
        else:
            return z_q_x, z_q_x_bar

    def indices_fetch(self, indices):
        indices_flatten = indices.reshape(indices.size(0) * indices.size(1))
        z_q_x_fetch = torch.index_select(self.embedding.weight, dim=0, index=indices_flatten).view(indices.size(0), indices.size(1), self.H, self.H)  # B*C*H*W

        return z_q_x_fetch

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, 2 * dim, 3, 1, 1),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(True),
            nn.Conv2d(2 * dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class VectorQuantizedVAE_Dim(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.dim = dim
        self.factor = 2
        #self.map = torch.nn.Parameter(torch.randn((1, dim, 4, 4)))
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 64 * self.factor, 4, 2, 1),
            nn.BatchNorm2d(64 * self.factor),
            nn.ReLU(True),
            #ResBlock(input_dim=64 * self.factor, latent_dim=64 * self.factor, output_dim=64 * self.factor),
            nn.Conv2d(64 * self.factor, 128 * self.factor, 4, 2, 1),
            nn.BatchNorm2d(128 * self.factor),
            nn.ReLU(True),
            #ResBlock(input_dim= 128 * self.factor, latent_dim= 128 * self.factor, output_dim=128 * self.factor),
            nn.Conv2d(128 * self.factor, 256 * self.factor, 4, 2, 1),
            nn.BatchNorm2d(256 * self.factor),
            nn.ReLU(True),
            nn.Conv2d(256 * self.factor, dim, 3, 1, 1),
            #ResBlock(input_dim=dim, latent_dim=dim,output_dim=dim),
            #ResBlock(input_dim=dim,latent_dim=dim,output_dim=dim)
            #ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, 4, 4)
        self.decoder = nn.Sequential(
            #ResBlock(dim),
            #ResBlock(input_dim=dim,latent_dim=dim,output_dim=dim),
            #nn.ReLU(True),
            nn.ConvTranspose2d(dim, 256 * self.factor, 3, 1, 1),
            nn.BatchNorm2d(256 * self.factor),
            nn.ReLU(True),
            nn.ConvTranspose2d(256 * self.factor, 128 * self.factor, 4, 2, 1),
            nn.BatchNorm2d(128 * self.factor),
            nn.ReLU(True),
            #ResBlock(input_dim=128 * self.factor, latent_dim=128 * self.factor, output_dim=128 * self.factor),
            nn.ConvTranspose2d(128 * self.factor, 64 * self.factor, 4, 2, 1),
            #ResBlock(input_dim=64 * self.factor, latent_dim=64 * self.factor, output_dim=64 * self.factor),
            nn.BatchNorm2d(64 * self.factor),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * self.factor, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        h = int(torch.sqrt(latents.size(2)))
        z_q_x = self.codebook.embedding(latents).resize(latents.size(0),latents.size(1),h,h)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x, full_sample=True, designate=None, index=False, interval=1):
        z_e_x = self.encoder(x)
        if not index:
            z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        else:
            z_q_x_st, z_q_x, indices = self.codebook.straight_through(z_e_x, index)
        area = z_e_x.size(1)
        if not full_sample:
            if designate is not None:
                sample_index = designate
            else:
                interval = int(interval)
                sample_index = (np.random.randint(area // interval) + 1) * interval
            z_q_x_st = torch.cat([
                z_q_x_st[:, :sample_index, :, :],
                z_q_x_st.new_zeros((z_q_x_st.size(0), area - sample_index, z_q_x_st.size(2), z_q_x_st.size(3)))
            ], dim=1)
            x_tilde = self.decoder(z_q_x_st)
            return x_tilde, z_e_x, z_q_x, sample_index
        else:
            x_tilde = self.decoder(z_q_x_st)
            if not index:
                return x_tilde, z_e_x, z_q_x
            else:
                return x_tilde, z_e_x, z_q_x, indices

    def indices_fetch(self, indices):
        z = self.codebook.indices_fetch(indices)
        if indices.size(1) < self.dim:
            zero_out = torch.zeros((indices.size(0), self.dim - indices.size(1), z.size(2), z.size(3))).cuda()
            z = torch.cat((z,zero_out),dim=1)
        x = self.decoder(z)
        return x

class VectorQuantizedVAE_CelebA(nn.Module):

    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.dim = dim
        self.factor = 2
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 64 * self.factor, 4, 2, 1),
            nn.BatchNorm2d(64 * self.factor),
            nn.ReLU(True),
            nn.Conv2d(64 * self.factor, 128 * self.factor, 4, 2, 1),
            nn.BatchNorm2d(128 * self.factor),
            nn.ReLU(True),
            nn.Conv2d(128 * self.factor, 256 * self.factor, 4, 2, 1),
            nn.BatchNorm2d(256 * self.factor),
            nn.ReLU(True),
            nn.Conv2d(256 * self.factor, dim, 3, 1, 1),
        )

        self.codebook = VQEmbedding(K, 8, 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, 256 * self.factor, 3, 1, 1),
            nn.BatchNorm2d(256 * self.factor),
            nn.ReLU(True),
            nn.ConvTranspose2d(256 * self.factor, 128 * self.factor, 4, 2, 1),
            nn.BatchNorm2d(128 * self.factor),
            nn.ReLU(True),
            nn.ConvTranspose2d(128 * self.factor, 64 * self.factor, 4, 2, 1),
            nn.BatchNorm2d(64 * self.factor),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * self.factor, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        h = int(torch.sqrt(latents.size(2)))
        z_q_x = self.codebook.embedding(latents).resize(latents.size(0),latents.size(1),h,h)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x, full_sample=True, designate=None, index=False, interval = 1):
        z_e_x = self.encoder(x)
        if not index:
            z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        else:
            z_q_x_st, z_q_x, indices = self.codebook.straight_through(z_e_x,index)
        area = z_e_x.size(1)
        if not full_sample:
            if designate != None:
                sample_index = designate
            else:
                interval = int(interval)
                sample_index = np.array((np.random.randint(area) // interval + 1) * interval)
                sample_index = sample_index.clip(1, area)
            zero_out = torch.zeros((z_q_x_st.size(0), area - sample_index, z_q_x_st.size(2), z_q_x_st.size(3))).cuda()
            z_q_x_st = torch.cat((z_q_x_st[:, :sample_index], zero_out), dim=1)
            x_tilde = self.decoder(z_q_x_st)
            return x_tilde, z_e_x, z_q_x, sample_index
        else:
            x_tilde = self.decoder(z_q_x_st)
            if not index:
                return x_tilde, z_e_x, z_q_x
            else:
                return x_tilde, z_e_x, z_q_x, indices

    def indices_fetch(self, indices):
        z = self.codebook.indices_fetch(indices)
        if indices.size(1) < self.dim:
            zero_out = torch.zeros((indices.size(0), self.dim - indices.size(1), z.size(2), z.size(3))).cuda()
            z = torch.cat((z,zero_out),dim=1)
        x = self.decoder(z)
        return x
class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)

class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()

        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h

class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)

    def forward(self, x, label):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def generate(self, label, shape=(7, 7), batch_size=64, sample_index=48):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )
        index_i = (sample_index // 7) + 1
        index_j = (sample_index % 7) + 1

        for i in range(index_i):
            if i < index_i - 1:
                for j in range(shape[1]):
                    logits = self.forward(x, label)
                    probs = F.softmax(logits[:, :, i, j], -1)
                    x.data[:, i, j].copy_(
                        probs.multinomial(1).squeeze().data
                    )
            else:
                for j in range(index_j):
                    logits = self.forward(x, label)
                    probs = F.softmax(logits[:, :, i, j], -1)
                    x.data[:, i, j].copy_(
                        probs.multinomial(1).squeeze().data
                    )
        return x


class VectorQuantizedVAE_Bigger(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.factor = 2
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, self.factor * 64, 4, 2, 1),
            nn.BatchNorm2d(self.factor * 64),
            nn.ReLU(True),
            nn.Conv2d(self.factor * 64, self.factor * 128, 4, 2, 1),
            # nn.BatchNorm2d(self.factor * 128),
            # nn.ReLU(True),
            # nn.Conv2d(self.factor * 128, self.factor * 256, 4, 2, 1),
            ResBlock(self.factor * 128),
            ResBlock(self.factor * 128),
            #ResBlock(self.factor * 256)
        )

        self.codebook = VQEmbedding_Original(K, self.factor * 128)
        #self.codebook = VQEmbedding(K, self.factor * 256)
        self.decoder = nn.Sequential(
            ResBlock(self.factor * 128),
            ResBlock(self.factor * 128),
            #ResBlock(self.factor * 256),
            # nn.ConvTranspose2d(self.factor * 256, self.factor * 128, 4, 2, 1),
            # nn.BatchNorm2d(self.factor * 128),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.factor * 128, self.factor * 64, 4, 2, 1),
            nn.BatchNorm2d(self.factor * 64),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.factor * 64, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0,3,1,2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x, index=False):
        z_e_x = self.encoder(x)
        if not index:
            z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
            x_tilde = self.decoder(z_q_x_st)
            return x_tilde, z_e_x, z_q_x
        else:
            z_q_x_st, z_q_x, indices = self.codebook.straight_through(z_e_x,index)
            x_tilde = self.decoder(z_q_x_st)
            return x_tilde, z_e_x, z_q_x, indices

    def indices_fetch(self, indices):
        z = self.codebook.indices_fetch(indices)
        if indices.size(1) < self.dim:
            zero_out = torch.zeros((indices.size(0), self.dim - indices.size(1), z.size(2), z.size(3))).cuda()
            z = torch.cat((z,zero_out),dim=1)
        x = self.decoder(z)
        return x

class LSTM(nn.Module):

    def __init__(self, k, embedding_dim=5000, hidden_size=2000,label_size = 100):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=k,
            embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=3, bias=True, batch_first=True)
        self.code_size = k
        self.fc = nn.Linear(hidden_size, self.code_size)

    def forward(self, input, label, hidden=None):
        embeddings = self.embedding(input.long())
        #label = label.unsqueeze(1).unsqueeze(2)
        #label = label.expand(embeddings.size(0), embeddings.size(1),1000).float()
        #embeddings = torch.cat((embeddings,label),dim=2)
        if hidden is None:
            lstm, (h, c) = self.lstm(embeddings)
        else:
            lstm, (h, c) = self.lstm(embeddings, hidden)
        lstm = lstm.contiguous().view(-1, lstm.shape[2])
        logits = self.fc(lstm)
        return logits, (h.detach(), c.detach())

    def sample(self, prior, seq_len, label):
        """
        Sample a string of length `seq_len` from the model.
        :param seq_len [int]: String length
        :param prior[tensor]: Prior of the first element
        :return [list]: A list of length `seq_len` that contains the index of each generated character.
        """

        with torch.no_grad():
            h_prev = None
            output = []
            x = torch.multinomial(prior, 1, replacement=True).type(torch.int64).cuda()
            x = x.unsqueeze(1)
            output.append(x.squeeze())
            for i in range(1, seq_len):

                logits, h_prev = self.forward(x, label, h_prev)
                np_logits = logits
                probs = torch.exp(np_logits)
                probs = torch.clamp(probs, 0, 1e30)
                x = torch.multinomial(probs, 1).type(torch.int64).cuda()
                # np_logits = logits[-1, :].to('cpu').numpy()
                # probs = np.exp(np_logits) / np.sum(np.exp(np_logits))
                # ix = np.random.choice(self.code_size, p=probs.ravel())
                # x = torch.tensor(ix, dtype=torch.int64)[None, None].cuda()
                output.append(x.squeeze())
        return output

    def sample_batch(self, prior, seq_len, batch_size, label):
        """
        Sample a string of length `seq_len` from the model.
        :param seq_len [int]: String length
        :return [list]: A list of length `seq_len` that contains the index of each generated character.
        """

        with torch.no_grad():
            h_prev = None
            output = torch.zeros((batch_size,seq_len)).cuda()
            x = torch.multinomial(prior, batch_size, replacement=True).type(torch.int64).cuda()
            x = x.unsqueeze(1)
            output[:, 0] = x.squeeze()
            for i in range(1, seq_len):
                logits, h_prev = self.forward(x, label, h_prev)
                np_logits = logits
                probs = torch.exp(np_logits)
                probs = torch.clamp(probs,0,1e30)
                x = torch.multinomial(probs, 1).type(torch.int64).cuda()
                output[:, i] = x.squeeze()
        return output