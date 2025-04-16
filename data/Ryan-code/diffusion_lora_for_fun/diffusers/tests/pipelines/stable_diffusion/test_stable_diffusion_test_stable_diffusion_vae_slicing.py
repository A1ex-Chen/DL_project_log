def test_stable_diffusion_vae_slicing(self):
    torch.cuda.reset_peak_memory_stats()
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    inputs = self.get_inputs(torch_device, dtype=torch.float16)
    inputs['prompt'] = [inputs['prompt']] * 4
    inputs['latents'] = torch.cat([inputs['latents']] * 4)
    image_sliced = pipe(**inputs).images
    mem_bytes = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    assert mem_bytes < 4000000000.0
    pipe.disable_vae_slicing()
    inputs = self.get_inputs(torch_device, dtype=torch.float16)
    inputs['prompt'] = [inputs['prompt']] * 4
    inputs['latents'] = torch.cat([inputs['latents']] * 4)
    image = pipe(**inputs).images
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes > 4000000000.0
    max_diff = numpy_cosine_similarity_distance(image_sliced.flatten(),
        image.flatten())
    assert max_diff < 0.01
