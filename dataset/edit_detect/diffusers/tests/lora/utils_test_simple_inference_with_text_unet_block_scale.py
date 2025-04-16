def test_simple_inference_with_text_unet_block_scale(self):
    """
        Tests a simple inference with lora attached to text encoder and unet, attaches
        one adapter and set differnt weights for different blocks (i.e. block lora)
        """
    for scheduler_cls in [DDIMScheduler, LCMScheduler]:
        components, text_lora_config, unet_lora_config = (self.
            get_dummy_components(scheduler_cls))
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)
        output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        pipe.text_encoder.add_adapter(text_lora_config, 'adapter-1')
        pipe.unet.add_adapter(unet_lora_config, 'adapter-1')
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder),
            'Lora not correctly set in text encoder')
        self.assertTrue(check_if_lora_correctly_set(pipe.unet),
            'Lora not correctly set in Unet')
        if self.has_two_text_encoders:
            pipe.text_encoder_2.add_adapter(text_lora_config, 'adapter-1')
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder_2
                ), 'Lora not correctly set in text encoder 2')
        weights_1 = {'text_encoder': 2, 'unet': {'down': 5}}
        pipe.set_adapters('adapter-1', weights_1)
        output_weights_1 = pipe(**inputs, generator=torch.manual_seed(0)
            ).images
        weights_2 = {'unet': {'up': 5}}
        pipe.set_adapters('adapter-1', weights_2)
        output_weights_2 = pipe(**inputs, generator=torch.manual_seed(0)
            ).images
        self.assertFalse(np.allclose(output_weights_1, output_weights_2,
            atol=0.001, rtol=0.001),
            'LoRA weights 1 and 2 should give different results')
        self.assertFalse(np.allclose(output_no_lora, output_weights_1, atol
            =0.001, rtol=0.001),
            'No adapter and LoRA weights 1 should give different results')
        self.assertFalse(np.allclose(output_no_lora, output_weights_2, atol
            =0.001, rtol=0.001),
            'No adapter and LoRA weights 2 should give different results')
        pipe.disable_lora()
        output_disabled = pipe(**inputs, generator=torch.manual_seed(0)).images
        self.assertTrue(np.allclose(output_no_lora, output_disabled, atol=
            0.001, rtol=0.001),
            'output with no lora and output with lora disabled should give same results'
            )
