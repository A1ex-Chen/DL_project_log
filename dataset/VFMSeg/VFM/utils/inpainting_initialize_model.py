def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt)['state_dict'], strict=False)
    device = torch.device('cuda') if torch.cuda.is_available(
        ) else torch.device('cpu')
    model = model.to(device)
    sampler = DDIMSampler(model)
    return sampler
