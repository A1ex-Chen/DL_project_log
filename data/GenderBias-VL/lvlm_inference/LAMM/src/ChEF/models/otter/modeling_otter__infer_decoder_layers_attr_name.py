def _infer_decoder_layers_attr_name(model: nn.Module):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]
    raise ValueError(
        f'We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually.'
        )
