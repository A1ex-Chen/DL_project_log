def test_stable_cascade_combined_prompt_embeds(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = StableCascadeCombinedPipeline(**components)
    pipe.set_progress_bar_config(disable=None)
    prompt = 'A photograph of a shiba inu, wearing a hat'
    (prompt_embeds, prompt_embeds_pooled, negative_prompt_embeds,
        negative_prompt_embeds_pooled) = pipe.prior_pipe.encode_prompt(device,
        1, 1, False, prompt=prompt)
    generator = torch.Generator(device=device)
    output_prompt = pipe(prompt=prompt, num_inference_steps=1,
        prior_num_inference_steps=1, output_type='np', generator=generator.
        manual_seed(0))
    output_prompt_embeds = pipe(prompt=None, prompt_embeds=prompt_embeds,
        prompt_embeds_pooled=prompt_embeds_pooled, negative_prompt_embeds=
        negative_prompt_embeds, negative_prompt_embeds_pooled=
        negative_prompt_embeds_pooled, num_inference_steps=1,
        prior_num_inference_steps=1, output_type='np', generator=generator.
        manual_seed(0))
    assert np.abs(output_prompt.images - output_prompt_embeds.images).max(
        ) < 1e-05
