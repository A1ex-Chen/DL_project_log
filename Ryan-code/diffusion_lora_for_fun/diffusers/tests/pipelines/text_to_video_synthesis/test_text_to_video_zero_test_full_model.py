def test_full_model(self):
    model_id = 'runwayml/stable-diffusion-v1-5'
    pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=
        torch.float16).to('cuda')
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    generator = torch.Generator(device='cuda').manual_seed(0)
    prompt = 'A bear is playing a guitar on Times Square'
    result = pipe(prompt=prompt, generator=generator).images
    expected_result = load_pt(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text-to-video/A bear is playing a guitar on Times Square.pt'
        )
    assert_mean_pixel_difference(result, expected_result)
