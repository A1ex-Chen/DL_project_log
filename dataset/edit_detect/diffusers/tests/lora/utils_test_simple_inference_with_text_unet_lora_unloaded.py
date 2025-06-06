def test_simple_inference_with_text_unet_lora_unloaded(self):
    """
        Tests a simple inference with lora attached to text encoder and unet, then unloads the lora weights
        and makes sure it works as expected
        """
    for scheduler_cls in [DDIMScheduler, LCMScheduler]:
        components, text_lora_config, unet_lora_config = (self.
            get_dummy_components(scheduler_cls))
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)
        output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(output_no_lora.shape == (1, 64, 64, 3))
        pipe.text_encoder.add_adapter(text_lora_config)
        pipe.unet.add_adapter(unet_lora_config)
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder),
            'Lora not correctly set in text encoder')
        self.assertTrue(check_if_lora_correctly_set(pipe.unet),
            'Lora not correctly set in Unet')
        if self.has_two_text_encoders:
            pipe.text_encoder_2.add_adapter(text_lora_config)
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder_2
                ), 'Lora not correctly set in text encoder 2')
        pipe.unload_lora_weights()
        self.assertFalse(check_if_lora_correctly_set(pipe.text_encoder),
            'Lora not correctly unloaded in text encoder')
        self.assertFalse(check_if_lora_correctly_set(pipe.unet),
            'Lora not correctly unloaded in Unet')
        if self.has_two_text_encoders:
            self.assertFalse(check_if_lora_correctly_set(pipe.
                text_encoder_2),
                'Lora not correctly unloaded in text encoder 2')
        ouput_unloaded = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(np.allclose(ouput_unloaded, output_no_lora, atol=
            0.001, rtol=0.001), 'Fused lora should change the output')
