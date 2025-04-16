def test_load_unload_load_kohya_lora(self):
    generator = torch.manual_seed(0)
    prompt = 'masterpiece, best quality, mountain'
    num_inference_steps = 2
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', safety_checker=None).to(torch_device)
    initial_images = pipe(prompt, output_type='np', generator=generator,
        num_inference_steps=num_inference_steps).images
    initial_images = initial_images[0, -3:, -3:, -1].flatten()
    lora_model_id = 'hf-internal-testing/civitai-colored-icons-lora'
    lora_filename = 'Colored_Icons_by_vizsumit.safetensors'
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    generator = torch.manual_seed(0)
    lora_images = pipe(prompt, output_type='np', generator=generator,
        num_inference_steps=num_inference_steps).images
    lora_images = lora_images[0, -3:, -3:, -1].flatten()
    pipe.unload_lora_weights()
    generator = torch.manual_seed(0)
    unloaded_lora_images = pipe(prompt, output_type='np', generator=
        generator, num_inference_steps=num_inference_steps).images
    unloaded_lora_images = unloaded_lora_images[0, -3:, -3:, -1].flatten()
    self.assertFalse(np.allclose(initial_images, lora_images))
    self.assertTrue(np.allclose(initial_images, unloaded_lora_images, atol=
        0.001))
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    generator = torch.manual_seed(0)
    lora_images_again = pipe(prompt, output_type='np', generator=generator,
        num_inference_steps=num_inference_steps).images
    lora_images_again = lora_images_again[0, -3:, -3:, -1].flatten()
    self.assertTrue(np.allclose(lora_images, lora_images_again, atol=0.001))
    release_memory(pipe)
