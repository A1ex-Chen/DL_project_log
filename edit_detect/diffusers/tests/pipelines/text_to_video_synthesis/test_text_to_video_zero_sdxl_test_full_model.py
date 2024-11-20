def test_full_model(self):
    model_id = 'stabilityai/stable-diffusion-xl-base-1.0'
    pipe = TextToVideoZeroSDXLPipeline.from_pretrained(model_id,
        torch_dtype=torch.float16, variant='fp16', use_safetensors=True)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    generator = torch.Generator(device='cpu').manual_seed(0)
    prompt = 'A panda dancing in Antarctica'
    result = pipe(prompt=prompt, generator=generator).images
    first_frame_slice = result[0, -3:, -3:, -1]
    last_frame_slice = result[-1, -3:, -3:, 0]
    expected_slice1 = np.array([0.57, 0.57, 0.57, 0.57, 0.57, 0.56, 0.55, 
        0.56, 0.56])
    expected_slice2 = np.array([0.54, 0.53, 0.53, 0.53, 0.53, 0.52, 0.53, 
        0.53, 0.53])
    assert np.abs(first_frame_slice.flatten() - expected_slice1).max() < 0.01
    assert np.abs(last_frame_slice.flatten() - expected_slice2).max() < 0.01
