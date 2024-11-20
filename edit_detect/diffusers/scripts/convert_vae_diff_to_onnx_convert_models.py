@torch.no_grad()
def convert_models(model_path: str, output_path: str, opset: int, fp16:
    bool=False):
    dtype = torch.float16 if fp16 else torch.float32
    if fp16 and torch.cuda.is_available():
        device = 'cuda'
    elif fp16 and not torch.cuda.is_available():
        raise ValueError(
            '`float16` model export is only supported on GPUs with CUDA')
    else:
        device = 'cpu'
    output_path = Path(output_path)
    vae_decoder = AutoencoderKL.from_pretrained(model_path + '/vae')
    vae_latent_channels = vae_decoder.config.latent_channels
    vae_decoder.forward = vae_decoder.decode
    onnx_export(vae_decoder, model_args=(torch.randn(1, vae_latent_channels,
        25, 25).to(device=device, dtype=dtype), False), output_path=
        output_path / 'vae_decoder' / 'model.onnx', ordered_input_names=[
        'latent_sample', 'return_dict'], output_names=['sample'],
        dynamic_axes={'latent_sample': {(0): 'batch', (1): 'channels', (2):
        'height', (3): 'width'}}, opset=opset)
    del vae_decoder
