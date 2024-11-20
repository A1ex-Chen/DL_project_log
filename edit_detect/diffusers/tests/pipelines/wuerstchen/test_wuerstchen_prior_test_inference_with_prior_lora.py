@require_peft_backend
def test_inference_with_prior_lora(self):
    _, prior_lora_config, _ = self.get_lora_components()
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    output_no_lora = pipe(**self.get_dummy_inputs(device))
    image_embed = output_no_lora.image_embeddings
    self.assertTrue(image_embed.shape == (1, 2, 24, 24))
    pipe.prior.add_adapter(prior_lora_config)
    self.assertTrue(self.check_if_lora_correctly_set(pipe.prior),
        'Lora not correctly set in prior')
    output_lora = pipe(**self.get_dummy_inputs(device))
    lora_image_embed = output_lora.image_embeddings
    self.assertTrue(image_embed.shape == lora_image_embed.shape)
