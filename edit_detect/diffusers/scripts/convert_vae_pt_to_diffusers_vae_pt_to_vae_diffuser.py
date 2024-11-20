def vae_pt_to_vae_diffuser(checkpoint_path: str, output_path: str):
    r = requests.get(
        ' https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml'
        )
    io_obj = io.BytesIO(r.content)
    original_config = yaml.safe_load(io_obj)
    image_size = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if checkpoint_path.endswith('safetensors'):
        from safetensors import safe_open
        checkpoint = {}
        with safe_open(checkpoint_path, framework='pt', device='cpu') as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)[
            'state_dict']
    vae_config = create_vae_diffusers_config(original_config, image_size=
        image_size)
    converted_vae_checkpoint = custom_convert_ldm_vae_checkpoint(checkpoint,
        vae_config)
    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    vae.save_pretrained(output_path)
