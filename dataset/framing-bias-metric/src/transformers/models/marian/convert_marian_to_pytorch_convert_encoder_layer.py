def convert_encoder_layer(opus_dict, layer_prefix: str, converter: dict):
    sd = {}
    for k in opus_dict:
        if not k.startswith(layer_prefix):
            continue
        stripped = remove_prefix(k, layer_prefix)
        v = opus_dict[k].T
        sd[converter[stripped]] = torch.tensor(v).squeeze()
    return sd
