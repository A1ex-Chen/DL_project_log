def test_stable_diffusion_negative_prompt_embeds(self):
    pipe = OnnxStableDiffusionPipeline.from_pretrained(self.hub_checkpoint,
        provider='CPUExecutionProvider')
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs()
    negative_prompt = 3 * ['this is a negative prompt']
    inputs['negative_prompt'] = negative_prompt
    inputs['prompt'] = 3 * [inputs['prompt']]
    output = pipe(**inputs)
    image_slice_1 = output.images[0, -3:, -3:, -1]
    inputs = self.get_dummy_inputs()
    prompt = 3 * [inputs.pop('prompt')]
    embeds = []
    for p in [prompt, negative_prompt]:
        text_inputs = pipe.tokenizer(p, padding='max_length', max_length=
            pipe.tokenizer.model_max_length, truncation=True,
            return_tensors='np')
        text_inputs = text_inputs['input_ids']
        embeds.append(pipe.text_encoder(input_ids=text_inputs.astype(np.
            int32))[0])
    inputs['prompt_embeds'], inputs['negative_prompt_embeds'] = embeds
    output = pipe(**inputs)
    image_slice_2 = output.images[0, -3:, -3:, -1]
    assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max(
        ) < 0.0001
