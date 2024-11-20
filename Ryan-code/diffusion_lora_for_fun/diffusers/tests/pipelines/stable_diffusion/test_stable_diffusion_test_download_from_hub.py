def test_download_from_hub(self):
    ckpt_paths = [
        'https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors'
        ,
        'https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors'
        ]
    for ckpt_path in ckpt_paths:
        pipe = StableDiffusionPipeline.from_single_file(ckpt_path,
            torch_dtype=torch.float16)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to('cuda')
    image_out = pipe('test', num_inference_steps=1, output_type='np').images[0]
    assert image_out.shape == (512, 512, 3)
