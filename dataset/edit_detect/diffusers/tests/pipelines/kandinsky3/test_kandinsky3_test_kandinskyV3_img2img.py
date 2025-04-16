def test_kandinskyV3_img2img(self):
    pipe = AutoPipelineForImage2Image.from_pretrained(
        'kandinsky-community/kandinsky-3', variant='fp16', torch_dtype=
        torch.float16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky3/t2i.png'
        )
    w, h = 512, 512
    image = image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    prompt = 'A painting of the inside of a subway train with tiny raccoons.'
    image = pipe(prompt, image=image, strength=0.75, num_inference_steps=5,
        generator=generator).images[0]
    assert image.size == (512, 512)
    expected_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky3/i2i.png'
        )
    image_processor = VaeImageProcessor()
    image_np = image_processor.pil_to_numpy(image)
    expected_image_np = image_processor.pil_to_numpy(expected_image)
    self.assertTrue(np.allclose(image_np, expected_image_np, atol=0.05))
