def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None
    if dropout > 0 and mask is None:
        if version.Version(torch.__version__) >= version.Version('1.2.0a'):
            mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()
        else:
            mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).byte()
    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask
    return mask, dropout
