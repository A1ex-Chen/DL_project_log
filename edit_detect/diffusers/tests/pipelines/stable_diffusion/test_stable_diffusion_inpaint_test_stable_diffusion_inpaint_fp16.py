def test_stable_diffusion_inpaint_fp16(self):
    vae = AsymmetricAutoencoderKL.from_pretrained(
        'cross-attention/asymmetric-autoencoder-kl-x-1-5', torch_dtype=
        torch.float16)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting', torch_dtype=torch.float16,
        safety_checker=None)
    pipe.unet.set_default_attn_processor()
    pipe.vae = vae
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(torch_device, dtype=torch.float16)
    image = pipe(**inputs).images
    image_slice = image[0, 253:256, 253:256, -1].flatten()
    assert image.shape == (1, 512, 512, 3)
    expected_slice = np.array([0.1343, 0.1406, 0.144, 0.1504, 0.1729, 
        0.0989, 0.1807, 0.2822, 0.1179])
    assert np.abs(expected_slice - image_slice).max() < 0.05
