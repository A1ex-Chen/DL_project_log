def check_quant_weight_correctness(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = {(k[len('module.'):] if k.startswith('module.') else k): v for
        k, v in state_dict.items()}
    quantizers_sd_keys = {f'{n[0]}._amax' for n in model.named_modules() if
        'quantizer' in n[0]}
    sd_all_keys = quantizers_sd_keys | set(model.state_dict().keys())
    assert set(state_dict.keys()
        ) == sd_all_keys, f'Passed quantized architecture, but following keys are missing in checkpoint: {list(sd_all_keys - set(state_dict.keys()))}'
