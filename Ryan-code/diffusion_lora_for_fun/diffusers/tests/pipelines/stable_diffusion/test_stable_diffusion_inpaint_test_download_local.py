def test_download_local(self):
    vae = AsymmetricAutoencoderKL.from_pretrained(
        'cross-attention/asymmetric-autoencoder-kl-x-1-5', torch_dtype=
        torch.float16)
    filename = hf_hub_download('runwayml/stable-diffusion-inpainting',
        filename='sd-v1-5-inpainting.ckpt')
    pipe = StableDiffusionInpaintPipeline.from_single_file(filename,
        torch_dtype=torch.float16)
    pipe.vae = vae
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to('cuda')
    inputs = self.get_inputs(torch_device)
    inputs['num_inference_steps'] = 1
    image_out = pipe(**inputs).images[0]
    assert image_out.shape == (512, 512, 3)
