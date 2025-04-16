def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model_path.split('/')[-1].split('.')[0]
    if not os.path.isfile(args.model_path):
        assert model_name == args.model_path, f'Make sure to provide one of the official model names {MODELS_MAP.keys()}'
        args.model_path = download(model_name)
    sample_rate = MODELS_MAP[model_name]['sample_rate']
    sample_size = MODELS_MAP[model_name]['sample_size']
    config = Object()
    config.sample_size = sample_size
    config.sample_rate = sample_rate
    config.latent_dim = 0
    diffusers_model = UNet1DModel(sample_size=sample_size, sample_rate=
        sample_rate)
    diffusers_state_dict = diffusers_model.state_dict()
    orig_model = DiffusionUncond(config)
    orig_model.load_state_dict(torch.load(args.model_path, map_location=
        device)['state_dict'])
    orig_model = orig_model.diffusion_ema.eval()
    orig_model_state_dict = orig_model.state_dict()
    renamed_state_dict = rename_orig_weights(orig_model_state_dict)
    renamed_minus_diffusers = set(renamed_state_dict.keys()) - set(
        diffusers_state_dict.keys())
    diffusers_minus_renamed = set(diffusers_state_dict.keys()) - set(
        renamed_state_dict.keys())
    assert len(renamed_minus_diffusers
        ) == 0, f'Problem with {renamed_minus_diffusers}'
    assert all(k.endswith('kernel') for k in list(diffusers_minus_renamed)
        ), f'Problem with {diffusers_minus_renamed}'
    for key, value in renamed_state_dict.items():
        assert diffusers_state_dict[key].squeeze().shape == value.squeeze(
            ).shape, f"Shape for {key} doesn't match. Diffusers: {diffusers_state_dict[key].shape} vs. {value.shape}"
        if key == 'time_proj.weight':
            value = value.squeeze()
        diffusers_state_dict[key] = value
    diffusers_model.load_state_dict(diffusers_state_dict)
    steps = 100
    seed = 33
    diffusers_scheduler = IPNDMScheduler(num_train_timesteps=steps)
    generator = torch.manual_seed(seed)
    noise = torch.randn([1, 2, config.sample_size], generator=generator).to(
        device)
    t = torch.linspace(1, 0, steps + 1, device=device)[:-1]
    step_list = get_crash_schedule(t)
    pipe = DanceDiffusionPipeline(unet=diffusers_model, scheduler=
        diffusers_scheduler)
    generator = torch.manual_seed(33)
    audio = pipe(num_inference_steps=steps, generator=generator).audios
    generated = sampling.iplms_sample(orig_model, noise, step_list, {})
    generated = generated.clamp(-1, 1)
    diff_sum = (generated - audio).abs().sum()
    diff_max = (generated - audio).abs().max()
    if args.save:
        pipe.save_pretrained(args.checkpoint_path)
    print('Diff sum', diff_sum)
    print('Diff max', diff_max)
    assert diff_max < 0.001, f'Diff max: {diff_max} is too much :-/'
    print(f'Conversion for {model_name} successful!')
