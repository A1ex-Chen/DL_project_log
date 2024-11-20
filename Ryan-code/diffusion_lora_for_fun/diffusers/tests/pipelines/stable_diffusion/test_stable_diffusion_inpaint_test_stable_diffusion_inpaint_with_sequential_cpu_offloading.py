def test_stable_diffusion_inpaint_with_sequential_cpu_offloading(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    vae = AsymmetricAutoencoderKL.from_pretrained(
        'cross-attention/asymmetric-autoencoder-kl-x-1-5', torch_dtype=
        torch.float16)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        'runwayml/stable-diffusion-inpainting', safety_checker=None,
        torch_dtype=torch.float16)
    pipe.vae = vae
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing(1)
    pipe.enable_sequential_cpu_offload()
    inputs = self.get_inputs(torch_device, dtype=torch.float16)
    _ = pipe(**inputs)
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes < 2.45 * 10 ** 9
