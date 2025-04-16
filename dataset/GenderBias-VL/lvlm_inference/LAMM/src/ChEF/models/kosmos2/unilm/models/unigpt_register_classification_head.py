def register_classification_head(self, name, num_classes=None, inner_dim=
    None, **kwargs):
    """Register a classification head."""
    if name in self.classification_heads:
        prev_num_classes = self.classification_heads[name
            ].out_proj.out_features
        prev_inner_dim = self.classification_heads[name].dense.out_features
        if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
            logger.warning(
                're-registering head "{}" with num_classes {} (prev: {}) and inner_dim {} (prev: {})'
                .format(name, num_classes, prev_num_classes, inner_dim,
                prev_inner_dim))
    self.classification_heads[name] = ClassificationHead(self.args.
        encoder_embed_dim, inner_dim or self.args.encoder_embed_dim,
        num_classes, self.args.pooler_activation_fn, self.args.
        pooler_dropout, self.args.ft_type)
