def test_simple_inference(self):
    """
        Tests a simple inference and makes sure it works as expected
        """
    for scheduler_cls in [DDIMScheduler, LCMScheduler]:
        components, text_lora_config, _ = self.get_dummy_components(
            scheduler_cls)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs()
        output_no_lora = pipe(**inputs).images
        self.assertTrue(output_no_lora.shape == (1, 64, 64, 3))
