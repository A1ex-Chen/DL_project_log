def build_model(ckpt_path=None, config=None, model_name='audioldm-s-full'):
    print('Load AudioLDM: %s', model_name)
    if ckpt_path is None:
        ckpt_path = get_metadata()[model_name]['path']
    if not os.path.exists(ckpt_path):
        download_checkpoint(model_name)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config(model_name)
    config['model']['params']['device'] = device
    config['model']['params']['cond_stage_key'] = 'text'
    latent_diffusion = LatentDiffusion(**config['model']['params'])
    resume_from_checkpoint = ckpt_path
    checkpoint = torch.load(resume_from_checkpoint, map_location=device)
    latent_diffusion.load_state_dict(checkpoint['state_dict'])
    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)
    latent_diffusion.cond_stage_model.embed_mode = 'text'
    return latent_diffusion
