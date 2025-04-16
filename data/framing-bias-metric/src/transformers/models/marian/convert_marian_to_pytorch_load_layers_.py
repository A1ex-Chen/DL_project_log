def load_layers_(layer_lst: torch.nn.ModuleList, opus_state: dict,
    converter, is_decoder=False):
    for i, layer in enumerate(layer_lst):
        layer_tag = (f'decoder_l{i + 1}_' if is_decoder else
            f'encoder_l{i + 1}_')
        sd = convert_encoder_layer(opus_state, layer_tag, converter)
        layer.load_state_dict(sd, strict=True)
