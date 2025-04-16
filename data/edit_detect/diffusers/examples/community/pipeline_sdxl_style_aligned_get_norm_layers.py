def get_norm_layers(pipeline_, norm_layers_: Dict[str, List[Union[nn.
    GroupNorm, nn.LayerNorm]]]):
    if isinstance(pipeline_, nn.LayerNorm) and share_layer_norm:
        norm_layers_['layer'].append(pipeline_)
    if isinstance(pipeline_, nn.GroupNorm) and share_group_norm:
        norm_layers_['group'].append(pipeline_)
    else:
        for layer in pipeline_.children():
            get_norm_layers(layer, norm_layers_)
