def test_img2img_pndm(self):
    sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5')
    sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    image = sd_pipe(**inputs).images[0]
    expected_image = load_numpy(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/stable_diffusion_1_5_pndm.npy'
        )
    max_diff = np.abs(expected_image - image).max()
    assert max_diff < 0.001
