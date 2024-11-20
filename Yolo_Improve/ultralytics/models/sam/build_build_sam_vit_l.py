def build_sam_vit_l(checkpoint=None):
    """Build and return a Segment Anything Model (SAM) l-size model."""
    return _build_sam(encoder_embed_dim=1024, encoder_depth=24,
        encoder_num_heads=16, encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint)
