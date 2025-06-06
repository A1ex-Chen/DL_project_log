@require_peft_version_greater(peft_version='0.6.2')
def test_simple_inference_with_text_lora_unet_fused_multi(self):
    """
        Tests a simple inference with lora attached into text encoder + fuses the lora weights into base model
        and makes sure it works as expected - with unet and multi-adapter case
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
        pipe.text_encoder.add_adapter(text_lora_config, 'adapter-1')
        pipe.unet.add_adapter(unet_lora_config, 'adapter-1')
        pipe.text_encoder.add_adapter(text_lora_config, 'adapter-2')
        pipe.unet.add_adapter(unet_lora_config, 'adapter-2')
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder),
            'Lora not correctly set in text encoder')
        self.assertTrue(check_if_lora_correctly_set(pipe.unet),
            'Lora not correctly set in Unet')
        if self.has_two_text_encoders:
            pipe.text_encoder_2.add_adapter(text_lora_config, 'adapter-1')
            pipe.text_encoder_2.add_adapter(text_lora_config, 'adapter-2')
            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder_2
                ), 'Lora not correctly set in text encoder 2')
        pipe.set_adapters(['adapter-1', 'adapter-2'])
        ouputs_all_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
        pipe.set_adapters(['adapter-1'])
        ouputs_lora_1 = pipe(**inputs, generator=torch.manual_seed(0)).images
        pipe.fuse_lora(adapter_names=['adapter-1'])
        outputs_lora_1_fused = pipe(**inputs, generator=torch.manual_seed(0)
            ).images
        self.assertTrue(np.allclose(ouputs_lora_1, outputs_lora_1_fused,
            atol=0.001, rtol=0.001), 'Fused lora should not change the output')
        pipe.unfuse_lora()
        pipe.fuse_lora(adapter_names=['adapter-2', 'adapter-1'])
        output_all_lora_fused = pipe(**inputs, generator=torch.manual_seed(0)
            ).images
        self.assertTrue(np.allclose(output_all_lora_fused, ouputs_all_lora,
            atol=0.001, rtol=0.001), 'Fused lora should not change the output')
