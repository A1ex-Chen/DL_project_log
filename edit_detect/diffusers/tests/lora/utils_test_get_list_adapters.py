def test_get_list_adapters(self):
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
        pipe.text_encoder.add_adapter(text_lora_config, 'adapter-1')
        pipe.unet.add_adapter(unet_lora_config, 'adapter-1')
        adapter_names = pipe.get_list_adapters()
        self.assertDictEqual(adapter_names, {'text_encoder': ['adapter-1'],
            'unet': ['adapter-1']})
        pipe.text_encoder.add_adapter(text_lora_config, 'adapter-2')
        pipe.unet.add_adapter(unet_lora_config, 'adapter-2')
        adapter_names = pipe.get_list_adapters()
        self.assertDictEqual(adapter_names, {'text_encoder': ['adapter-1',
            'adapter-2'], 'unet': ['adapter-1', 'adapter-2']})
        pipe.set_adapters(['adapter-1', 'adapter-2'])
        self.assertDictEqual(pipe.get_list_adapters(), {'unet': [
            'adapter-1', 'adapter-2'], 'text_encoder': ['adapter-1',
            'adapter-2']})
        pipe.unet.add_adapter(unet_lora_config, 'adapter-3')
        self.assertDictEqual(pipe.get_list_adapters(), {'unet': [
            'adapter-1', 'adapter-2', 'adapter-3'], 'text_encoder': [
            'adapter-1', 'adapter-2']})
