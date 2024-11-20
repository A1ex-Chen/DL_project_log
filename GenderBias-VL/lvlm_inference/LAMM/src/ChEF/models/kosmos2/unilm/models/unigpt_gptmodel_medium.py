@register_model_architecture('unigptmodel', 'unigptmodel_medium')
def gptmodel_medium(args):
    args.decoder_layers = safe_getattr(args, 'decoder_layers', 24)
    args.decoder_embed_dim = safe_getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_attention_heads = safe_getattr(args,
        'decoder_attention_heads', 16)
    args.decoder_learned_pos = safe_getattr(args, 'decoder_learned_pos', False)
    args.pooler_dropout = safe_getattr(args, 'pooler_dropout', 0.1)
    base_gpt3_architecture(args)
    roberta_base_architecture(args)
