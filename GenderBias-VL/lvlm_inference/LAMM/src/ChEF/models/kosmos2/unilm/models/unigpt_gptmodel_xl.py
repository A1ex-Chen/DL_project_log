@register_model_architecture('unigptmodel', 'unigptmodel_xl')
def gptmodel_xl(args):
    args.decoder_layers = safe_getattr(args, 'decoder_layers', 24)
    args.decoder_embed_dim = safe_getattr(args, 'decoder_embed_dim', 2048)
    args.decoder_attention_heads = safe_getattr(args,
        'decoder_attention_heads', 32)
    args.decoder_learned_pos = safe_getattr(args, 'decoder_learned_pos', False)
    args.dropout = safe_getattr(args, 'dropout', 0.1)
    args.attention_dropout = safe_getattr(args, 'attention_dropout', 0.1)
    base_gpt3_architecture(args)
    roberta_large_architecture(args)
