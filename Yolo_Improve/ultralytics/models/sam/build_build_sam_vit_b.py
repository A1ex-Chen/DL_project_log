def build_sam_vit_b(checkpoint=None):
    """Build and return a Segment Anything Model (SAM) b-size model."""
    return _build_sam(encoder_embed_dim=768, encoder_depth=12,
        encoder_num_heads=12, encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint)
