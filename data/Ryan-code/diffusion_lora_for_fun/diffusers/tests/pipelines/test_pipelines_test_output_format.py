def test_output_format(self):
    model_path = 'google/ddpm-cifar10-32'
    scheduler = DDIMScheduler.from_pretrained(model_path)
    pipe = DDIMPipeline.from_pretrained(model_path, scheduler=scheduler)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    images = pipe(output_type='np').images
    assert images.shape == (1, 32, 32, 3)
    assert isinstance(images, np.ndarray)
    images = pipe(output_type='pil', num_inference_steps=4).images
    assert isinstance(images, list)
    assert len(images) == 1
    assert isinstance(images[0], PIL.Image.Image)
    images = pipe(num_inference_steps=4).images
    assert isinstance(images, list)
    assert isinstance(images[0], PIL.Image.Image)
