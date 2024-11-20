def test_simple_inference_with_text_unet_lora_save_load(self):
    """
        Tests a simple usecase where users could use saving utilities for LoRA for Unet + text encoder
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
        images_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        with tempfile.TemporaryDirectory() as tmpdirname:
            text_encoder_state_dict = get_peft_model_state_dict(pipe.
                text_encoder)
            unet_state_dict = get_peft_model_state_dict(pipe.unet)
            if self.has_two_text_encoders:
                text_encoder_2_state_dict = get_peft_model_state_dict(pipe.
                    text_encoder_2)
                self.pipeline_class.save_lora_weights(save_directory=
                    tmpdirname, text_encoder_lora_layers=
                    text_encoder_state_dict, text_encoder_2_lora_layers=
                    text_encoder_2_state_dict, unet_lora_layers=
                    unet_state_dict, safe_serialization=False)
            else:
                self.pipeline_class.save_lora_weights(save_directory=
                    tmpdirname, text_encoder_lora_layers=
                    text_encoder_state_dict, unet_lora_layers=
                    unet_state_dict, safe_serialization=False)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname,
                'pytorch_lora_weights.bin')))
            pipe.unload_lora_weights()
            pipe.load_lora_weights(os.path.join(tmpdirname,
                'pytorch_lora_weights.bin'))
        images_lora_from_pretrained = pipe(**inputs, generator=torch.
            manual_seed(0)).images
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder),
            'Lora not correctly set in text encoder')
        self.assertTrue(check_if_lora_correctly_set(pipe.unet),
            'Lora not correctly set in Unet')
        if self.has_two_text_encoders:
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder_2
                ), 'Lora not correctly set in text encoder 2')
        self.assertTrue(np.allclose(images_lora,
            images_lora_from_pretrained, atol=0.001, rtol=0.001),
            'Loading from saved checkpoints should give same results.')
