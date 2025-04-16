def test_simple_inference_save_pretrained(self):
    """
        Tests a simple usecase where users could use saving utilities for LoRA through save_pretrained
        """
    for scheduler_cls in [DDIMScheduler, LCMScheduler]:
        components, text_lora_config, _ = self.get_dummy_components(
            scheduler_cls)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)
        output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(output_no_lora.shape == (1, 64, 64, 3))
        pipe.text_encoder.add_adapter(text_lora_config)
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder),
            'Lora not correctly set in text encoder')
        if self.has_two_text_encoders:
            pipe.text_encoder_2.add_adapter(text_lora_config)
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder_2
                ), 'Lora not correctly set in text encoder 2')
        images_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe_from_pretrained = self.pipeline_class.from_pretrained(
                tmpdirname)
            pipe_from_pretrained.to(torch_device)
        self.assertTrue(check_if_lora_correctly_set(pipe_from_pretrained.
            text_encoder), 'Lora not correctly set in text encoder')
        if self.has_two_text_encoders:
            self.assertTrue(check_if_lora_correctly_set(
                pipe_from_pretrained.text_encoder_2),
                'Lora not correctly set in text encoder 2')
        images_lora_save_pretrained = pipe_from_pretrained(**inputs,
            generator=torch.manual_seed(0)).images
        self.assertTrue(np.allclose(images_lora,
            images_lora_save_pretrained, atol=0.001, rtol=0.001),
            'Loading from saved checkpoints should give same results.')
