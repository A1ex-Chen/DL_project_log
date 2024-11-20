def test_prompt_embeds(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.set_progress_bar_config(disable=None)
    pipe.to(torch_device)
    inputs = self.get_dummy_inputs(torch_device)
    prompt = inputs.pop('prompt')
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
        negative_pooled_prompt_embeds) = pipe.encode_prompt(prompt)
    pipe(**inputs, prompt_embeds=prompt_embeds, negative_prompt_embeds=
        negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds)
