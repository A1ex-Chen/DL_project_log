def test_depth2img_pndm(self):
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-depth')
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs()
    image = pipe(**inputs).images[0]
    expected_image = load_numpy(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_depth2img/stable_diffusion_2_0_pndm.npy'
        )
    max_diff = np.abs(expected_image - image).max()
    assert max_diff < 0.001
