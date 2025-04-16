def _get_base_config(img_size=224, patch_size=16, drop_path_rate=0,
    checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs):
    return EncoderConfig(img_size=img_size, patch_size=patch_size,
        vocab_size=vocab_size, multiway=True, layernorm_embedding=False,
        normalize_output=True, no_output_layer=True, drop_path_rate=
        drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12,
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12,
        checkpoint_activations=checkpoint_activations)
