def test_stable_diffusion_text2img_pipeline_v_pred_fp16(self):
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-text2img/astronaut_riding_a_horse_v_pred_fp16.npy'
        )
    pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2', torch_dtype=torch.float16)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    prompt = 'astronaut riding a horse'
    generator = torch.manual_seed(0)
    output = pipe(prompt=prompt, guidance_scale=7.5, generator=generator,
        output_type='np')
    image = output.images[0]
    assert image.shape == (768, 768, 3)
    max_diff = numpy_cosine_similarity_distance(image.flatten(),
        expected_image.flatten())
    assert max_diff < 0.001
