def build_anchors(cfg):
    anchor_generator = build_anchor_generator(cfg)
    anchor_labeler = build_anchor_labeler(cfg, anchor_generator)
    return anchor_generator, anchor_labeler
