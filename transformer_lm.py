
from fairseq import options, utils
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    Embedding,
    TransformerDecoder,
)
from fairseq.modules import (
    AdaptiveInput,
    CharacterTokenEmbedder,
)

DEFAULT_MAX_TARGET_POSITIONS = 1024


#@register_model('transformer_lm')
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
            'transformer_lm.gbw.adaptive_huge': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2',
            'transformer_lm.wiki103.adaptive': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.tar.bz2',
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

        parser.add_argument('--no-decoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        # fmt: on

    @classmethod
    def build_model(cls, args, vocab_size):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_lm_architecture(args)
        embed_tokens = Embedding(vocab_size, args.decoder_input_dim)
        decoder = TransformerDecoder(
            args, None, embed_tokens, no_encoder_attn=True,
        )
        return TransformerLanguageModel(decoder)

#@register_model_architecture('transformer_lm', 'transformer_lm')
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, 'decoder_final_norm'):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = 0.1
    args.attention_dropout = 0.0
    args.activation_dropout = 0.0
    args.decoder_embed_dim = 512
    args.decoder_ffn_embed_dim = 2048
    args.decoder_layers = 6
    args.decoder_attention_heads = 8
    args.decoder_learned_pos = False
    args.activation_fn = 'relu'

    args.add_bos_token = False
    args.no_token_positional_embeddings = False
    args.share_decoder_input_output_embed = True
    args.decoder_output_dim = 512
    args.decoder_input_dim = 512

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = False
    args.no_scale_embedding = False
    args.layernorm_embedding = False
    args.max_target_positions = 100
