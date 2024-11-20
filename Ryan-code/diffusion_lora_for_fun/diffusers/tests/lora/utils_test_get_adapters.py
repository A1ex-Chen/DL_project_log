def test_get_adapters(self):
    """
        Tests a simple usecase where we attach multiple adapters and check if the results
        are the expected results
        """
    for scheduler_cls in [DDIMScheduler, LCMScheduler]:
        components, text_lora_config, unet_lora_config = (self.
            get_dummy_components(scheduler_cls))
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)
        pipe.text_encoder.add_adapter(text_lora_config, 'adapter-1')
        pipe.unet.add_adapter(unet_lora_config, 'adapter-1')
        adapter_names = pipe.get_active_adapters()
        self.assertListEqual(adapter_names, ['adapter-1'])
        pipe.text_encoder.add_adapter(text_lora_config, 'adapter-2')
        pipe.unet.add_adapter(unet_lora_config, 'adapter-2')
        adapter_names = pipe.get_active_adapters()
        self.assertListEqual(adapter_names, ['adapter-2'])
        pipe.set_adapters(['adapter-1', 'adapter-2'])
        self.assertListEqual(pipe.get_active_adapters(), ['adapter-1',
            'adapter-2'])
