def test_simple_inference_with_text_lora_fused(self):
    """
        Tests a simple inference with lora attached into text encoder + fuses the lora weights into base model
        and makes sure it works as expected
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
        pipe.fuse_lora()
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder),
            'Lora not correctly set in text encoder')
        if self.has_two_text_encoders:
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder_2
                ), 'Lora not correctly set in text encoder 2')
        ouput_fused = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertFalse(np.allclose(ouput_fused, output_no_lora, atol=
            0.001, rtol=0.001), 'Fused lora should change the output')
