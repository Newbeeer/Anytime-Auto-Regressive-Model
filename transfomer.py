import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        # self.list_layer = nn.ModuleList(
        #     [TransformerEncoderLayer(ninp, nhead, nhid, dropout) for i in range(nlayers)]
        # )

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        '''

        :param sz:
        :return: mask : mask[i,j]:
        '''
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def sample(self, prior, seq_len, label):

        with torch.no_grad():
            output = []
            input = torch.multinomial(prior, 1, replacement=True).type(torch.int64).cuda()
            input = input.unsqueeze(0) # (1, batch_size)
            output.append(input.squeeze())
            for i in range(1, seq_len):
                logits = self.forward(input)[-1].squeeze() # last entry in the sequence : (Batch_size * Class)
                logits = logits.exp()
                x = torch.multinomial(logits, 1).type(torch.int64).cuda()
                output.append(x.squeeze())
                input = torch.cat([input, x.unsqueeze(0)], 0)  # Enlarge the seq_len (first dimenstion)
        return output

    def sample_batch(self, prior, seq_len, batch_size, label):

        with torch.no_grad():
            output = torch.zeros((batch_size, seq_len)).cuda()
            input = torch.multinomial(prior, batch_size, replacement=True).type(torch.int64).cuda()
            input = input.unsqueeze(0)  #(1, batch_size)
            output[:, 0] = input.squeeze()
            for i in range(1, seq_len):
                logits = self.forward(input)[-1].squeeze() # last entry in the sequence : (Batch_size * Class)
                logits = logits.exp()
                x = torch.multinomial(logits, 1).type(torch.int64).cuda()
                output[:, i] = x.squeeze()
                input = torch.cat([input, x.transpose(0, 1)], 0)  # Enlarge the seq_len (first dimenstion)

        return output
