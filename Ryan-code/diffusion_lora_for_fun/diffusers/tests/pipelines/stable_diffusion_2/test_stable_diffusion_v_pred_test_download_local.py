def test_download_local(self):
    filename = hf_hub_download('stabilityai/stable-diffusion-2-1', filename
        ='v2-1_768-ema-pruned.safetensors')
    pipe = StableDiffusionPipeline.from_single_file(filename, torch_dtype=
        torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    image_out = pipe('test', num_inference_steps=1, output_type='np').images[0]
    assert image_out.shape == (768, 768, 3)
