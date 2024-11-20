def _load_grounding_dino_model(self, model_config_path,
    model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    load_res = model.load_state_dict(clean_state_dict(checkpoint['model']),
        strict=False)
    print(load_res)
    model = model.to(device)
    _ = model.eval()
    return model
