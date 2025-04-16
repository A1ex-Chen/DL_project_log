def get_sd_vae_model(self, model_id=
    'cross-attention/asymmetric-autoencoder-kl-x-1-5', fp16=False):
    revision = 'main'
    torch_dtype = torch.float32
    model = AsymmetricAutoencoderKL.from_pretrained(model_id, torch_dtype=
        torch_dtype, revision=revision)
    model.to(torch_device).eval()
    return model
