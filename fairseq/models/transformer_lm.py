# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import utils
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    Embedding,
    TransformerDecoder,
)

DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('transformer_lm')
class TransformerLanguageModel(FairseqLanguageModel):

    @classmethod
    def hub_models(cls):

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer_lm.wmt19.en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.bz2'),
            'transformer_lm.wmt19.de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.bz2'),
            'transformer_lm.wmt19.ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.bz2'),
        }

    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension')
        parser.add_argument('--decoder-input-dim', type=int, metavar='N',
                            help='decoder input dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-decoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last decoder block')
        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = getattr(args, 'tokens_per_sample', DEFAULT_MAX_TARGET_POSITIONS)

        embed_tokens = Embedding(len(task.source_dictionary), args.decoder_input_dim, task.source_dictionary.pad())

        decoder = TransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True,
        )
        return TransformerLanguageModel(decoder)


@register_model_architecture('transformer_lm', 'transformer_lm')
def base_lm_architecture(args):
    if hasattr(args, 'decoder_final_norm'):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', True)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')

    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', False)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', True)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', True)


@register_model_architecture('transformer_lm', 'transformer_lm_L6A12')
def transformer_lm_l6a12(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    base_lm_architecture(args)


@register_model_architecture('transformer_lm', 'transformer_lm_L5A12')
def transformer_lm_l5a12(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_layers = getattr(args, 'decoder_layers', 5)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    base_lm_architecture(args)


@register_model_architecture('transformer_lm', 'transformer_lm_L4A12')
def transformer_lm_l4a12(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    base_lm_architecture(args)


@register_model_architecture('transformer_lm', 'transformer_lm_L3A12')
def transformer_lm_l3a12(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    base_lm_architecture(args)


@register_model_architecture('transformer_lm', 'transformer_lm_L5A8')
def transformer_lm_l5a8(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 5)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    base_lm_architecture(args)


@register_model_architecture('transformer_lm', 'transformer_lm_L4A8')
def transformer_lm_l4a8(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 4)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    base_lm_architecture(args)


@register_model_architecture('transformer_lm', 'transformer_lm_L6A10')
def transformer_lm_l6a10(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 640)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2560)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 10)
    base_lm_architecture(args)


@register_model_architecture('transformer_lm', 'transformer_lm_L5A10')
def transformer_lm_l5a10(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 640)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2560)
    args.decoder_layers = getattr(args, 'decoder_layers', 5)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 10)
    base_lm_architecture(args)
