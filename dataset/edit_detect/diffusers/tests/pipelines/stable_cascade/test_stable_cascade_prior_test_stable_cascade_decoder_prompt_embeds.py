def test_stable_cascade_decoder_prompt_embeds(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.set_progress_bar_config(disable=None)
    prompt = 'A photograph of a shiba inu, wearing a hat'
    (prompt_embeds, prompt_embeds_pooled, negative_prompt_embeds,
        negative_prompt_embeds_pooled) = pipe.encode_prompt(device, 1, 1, 
        False, prompt=prompt)
    generator = torch.Generator(device=device)
    output_prompt = pipe(prompt=prompt, num_inference_steps=1, output_type=
        'np', generator=generator.manual_seed(0))
    output_prompt_embeds = pipe(prompt=None, prompt_embeds=prompt_embeds,
        prompt_embeds_pooled=prompt_embeds_pooled, negative_prompt_embeds=
        negative_prompt_embeds, negative_prompt_embeds_pooled=
        negative_prompt_embeds_pooled, num_inference_steps=1, output_type=
        'np', generator=generator.manual_seed(0))
    assert np.abs(output_prompt.image_embeddings - output_prompt_embeds.
        image_embeddings).max() < 1e-05
