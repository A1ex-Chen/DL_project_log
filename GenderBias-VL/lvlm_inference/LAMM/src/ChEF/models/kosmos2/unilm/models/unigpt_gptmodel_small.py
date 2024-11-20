@register_model_architecture('unigptmodel', 'unigptmodel_small')
def gptmodel_small(args):
    args.decoder_layers = safe_getattr(args, 'decoder_layers', 12)
    args.decoder_embed_dim = safe_getattr(args, 'decoder_embed_dim', 768)
    args.decoder_attention_heads = safe_getattr(args,
        'decoder_attention_heads', 12)
    args.decoder_learned_pos = safe_getattr(args, 'decoder_learned_pos', False)
    base_gpt3_architecture(args)
    roberta_base_architecture(args)
