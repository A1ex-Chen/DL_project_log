@require_compel
def test_weighted_prompts_compel(self):
    from compel import Compel
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4')
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    prompt = 'a red cat playing with a ball{}'
    prompts = [prompt.format(s) for s in ['', '++', '--']]
    prompt_embeds = compel(prompts)
    generator = [torch.Generator(device='cpu').manual_seed(33) for _ in
        range(prompt_embeds.shape[0])]
    images = pipe(prompt_embeds=prompt_embeds, generator=generator,
        num_inference_steps=20, output_type='np').images
    for i, image in enumerate(images):
        expected_image = load_numpy(
            f'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/forest_{i}.npy'
            )
        assert np.abs(image - expected_image).max() < 0.3
