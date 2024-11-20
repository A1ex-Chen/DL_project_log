def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    orig_config_path = huggingface_hub.hf_hub_download(UPSCALER_REPO,
        'config_laion_text_cond_latent_upscaler_2.json')
    orig_weights_path = huggingface_hub.hf_hub_download(UPSCALER_REPO,
        'laion_text_cond_latent_upscaler_2_1_00470000_slim.pth')
    print(f'loading original model configuration from {orig_config_path}')
    print(f'loading original model checkpoint from {orig_weights_path}')
    print('converting to diffusers unet')
    orig_config = K.config.load_config(open(orig_config_path))['model']
    model = unet_model_from_original_config(orig_config)
    orig_checkpoint = torch.load(orig_weights_path, map_location=device)[
        'model_ema']
    converted_checkpoint = unet_to_diffusers_checkpoint(model, orig_checkpoint)
    model.load_state_dict(converted_checkpoint, strict=True)
    model.save_pretrained(args.dump_path)
    print(f'saving converted unet model in {args.dump_path}')
