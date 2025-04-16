def test_kandinsky_inpaint(self):
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/kandinskyv22_inpaint_cat_with_hat_fp16.npy'
        )
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png'
        )
    mask = np.zeros((768, 768), dtype=np.float32)
    mask[:250, 250:-250] = 1
    prompt = 'a hat'
    pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-prior', torch_dtype=torch.float16)
    pipe_prior.to(torch_device)
    pipeline = KandinskyV22InpaintPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-decoder-inpaint', torch_dtype=
        torch.float16)
    pipeline = pipeline.to(torch_device)
    pipeline.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    image_emb, zero_image_emb = pipe_prior(prompt, generator=generator,
        num_inference_steps=2, negative_prompt='').to_tuple()
    generator = torch.Generator(device='cpu').manual_seed(0)
    output = pipeline(image=init_image, mask_image=mask, image_embeds=
        image_emb, negative_image_embeds=zero_image_emb, generator=
        generator, num_inference_steps=2, height=768, width=768,
        output_type='np')
    image = output.images[0]
    assert image.shape == (768, 768, 3)
    max_diff = numpy_cosine_similarity_distance(expected_image.flatten(),
        image.flatten())
    assert max_diff < 0.0001
