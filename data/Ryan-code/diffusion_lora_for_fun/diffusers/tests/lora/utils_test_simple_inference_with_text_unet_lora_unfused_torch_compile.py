@unittest.skip('This is failing for now - need to investigate')
def test_simple_inference_with_text_unet_lora_unfused_torch_compile(self):
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
        pipe.unet = torch.compile(pipe.unet, mode='reduce-overhead',
            fullgraph=True)
        pipe.text_encoder = torch.compile(pipe.text_encoder, mode=
            'reduce-overhead', fullgraph=True)
        if self.has_two_text_encoders:
            pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, mode=
                'reduce-overhead', fullgraph=True)
        _ = pipe(**inputs, generator=torch.manual_seed(0)).images
