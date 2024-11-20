def test_text_to_image_model_cpu_offload(self):
    image_encoder = self.get_image_encoder(repo_id='h94/IP-Adapter',
        subfolder='models/image_encoder')
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', image_encoder=image_encoder,
        safety_checker=None, torch_dtype=self.dtype)
    pipeline.load_ip_adapter('h94/IP-Adapter', subfolder='models',
        weight_name='ip-adapter_sd15.bin')
    pipeline.to(torch_device)
    inputs = self.get_dummy_inputs()
    output_without_offload = pipeline(**inputs).images
    pipeline.enable_model_cpu_offload()
    inputs = self.get_dummy_inputs()
    output_with_offload = pipeline(**inputs).images
    max_diff = np.abs(output_with_offload - output_without_offload).max()
    self.assertLess(max_diff, 0.001,
        'CPU offloading should not affect the inference results')
    offloaded_modules = [v for k, v in pipeline.components.items() if 
        isinstance(v, torch.nn.Module) and k not in pipeline.
        _exclude_from_cpu_offload]
    self.assertTrue(all(v.device.type == 'cpu' for v in offloaded_modules)
        ), f"Not offloaded: {[v for v in offloaded_modules if v.device.type != 'cpu']}"
