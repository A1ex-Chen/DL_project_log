@skip_mps
def test_lora_fuse_nan(self):
    for scheduler_cls in [DDIMScheduler, LCMScheduler]:
        components, text_lora_config, unet_lora_config = (self.
            get_dummy_components(scheduler_cls))
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)
        pipe.text_encoder.add_adapter(text_lora_config, 'adapter-1')
        pipe.unet.add_adapter(unet_lora_config, 'adapter-1')
        self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder),
            'Lora not correctly set in text encoder')
        self.assertTrue(check_if_lora_correctly_set(pipe.unet),
            'Lora not correctly set in Unet')
        with torch.no_grad():
            pipe.unet.mid_block.attentions[0].transformer_blocks[0
                ].attn1.to_q.lora_A['adapter-1'].weight += float('inf')
        with self.assertRaises(ValueError):
            pipe.fuse_lora(safe_fusing=True)
        pipe.fuse_lora(safe_fusing=False)
        out = pipe('test', num_inference_steps=2, output_type='np').images
        self.assertTrue(np.isnan(out).all())
