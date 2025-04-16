def test_download_local(self):
    ckpt_filename = hf_hub_download('runwayml/stable-diffusion-v1-5',
        filename='v1-5-pruned-emaonly.safetensors')
    config_filename = hf_hub_download('runwayml/stable-diffusion-v1-5',
        filename='v1-inference.yaml')
    pipe = StableDiffusionPipeline.from_single_file(ckpt_filename,
        config_files={'v1': config_filename}, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to('cuda')
    image_out = pipe('test', num_inference_steps=1, output_type='np').images[0]
    assert image_out.shape == (512, 512, 3)
