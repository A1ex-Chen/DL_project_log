def test_kandinskyV3(self):
    pipe = AutoPipelineForText2Image.from_pretrained(
        'kandinsky-community/kandinsky-3', variant='fp16', torch_dtype=
        torch.float16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    prompt = (
        'A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background.'
        )
    generator = torch.Generator(device='cpu').manual_seed(0)
    image = pipe(prompt, num_inference_steps=5, generator=generator).images[0]
    assert image.size == (1024, 1024)
    expected_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky3/t2i.png'
        )
    image_processor = VaeImageProcessor()
    image_np = image_processor.pil_to_numpy(image)
    expected_image_np = image_processor.pil_to_numpy(expected_image)
    self.assertTrue(np.allclose(image_np, expected_image_np, atol=0.05))
