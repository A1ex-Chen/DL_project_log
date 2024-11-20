def test_kandinsky_text2img(self):
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/kandinsky_text2img_cat_fp16.npy'
        )
    pipe_prior = KandinskyPriorPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-1-prior', torch_dtype=torch.float16)
    pipe_prior.to(torch_device)
    pipeline = KandinskyPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-1', torch_dtype=torch.float16)
    pipeline.to(torch_device)
    pipeline.set_progress_bar_config(disable=None)
    prompt = 'red cat, 4k photo'
    generator = torch.Generator(device='cuda').manual_seed(0)
    image_emb, zero_image_emb = pipe_prior(prompt, generator=generator,
        num_inference_steps=5, negative_prompt='').to_tuple()
    generator = torch.Generator(device='cuda').manual_seed(0)
    output = pipeline(prompt, image_embeds=image_emb, negative_image_embeds
        =zero_image_emb, generator=generator, num_inference_steps=100,
        output_type='np')
    image = output.images[0]
    assert image.shape == (512, 512, 3)
    assert_mean_pixel_difference(image, expected_image)
