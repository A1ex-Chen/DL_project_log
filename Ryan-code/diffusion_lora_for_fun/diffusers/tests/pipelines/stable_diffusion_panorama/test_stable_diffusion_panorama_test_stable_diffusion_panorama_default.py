def test_stable_diffusion_panorama_default(self):
    model_ckpt = 'stabilityai/stable-diffusion-2-base'
    scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder='scheduler'
        )
    pipe = StableDiffusionPanoramaPipeline.from_pretrained(model_ckpt,
        scheduler=scheduler, safety_checker=None)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs()
    image = pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1].flatten()
    assert image.shape == (1, 512, 2048, 3)
    expected_slice = np.array([0.36968392, 0.27025372, 0.32446766, 
        0.28379387, 0.36363274, 0.30733347, 0.27100027, 0.27054125, 0.25536096]
        )
    assert np.abs(expected_slice - image_slice).max() < 0.01
