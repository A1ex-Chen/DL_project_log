@require_peft_version_greater(peft_version='0.9.0')
def test_simple_inference_with_dora(self):
    for scheduler_cls in [DDIMScheduler, LCMScheduler]:
        components, text_lora_config, unet_lora_config = (self.
            get_dummy_components(scheduler_cls, use_dora=True))
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)
        output_no_dora_lora = pipe(**inputs, generator=torch.manual_seed(0)
            ).images
        self.assertTrue(output_no_dora_lora.shape == (1, 64, 64, 3))
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
        output_dora_lora = pipe(**inputs, generator=torch.manual_seed(0)
            ).images
        self.assertFalse(np.allclose(output_dora_lora, output_no_dora_lora,
            atol=0.001, rtol=0.001), 'DoRA lora should change the output')
