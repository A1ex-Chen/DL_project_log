def test_ldm_default_ddim(self):
    pipe = LDMTextToImagePipeline.from_pretrained(
        'CompVis/ldm-text2im-large-256').to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    image = pipe(**inputs).images[0]
    expected_image = load_numpy(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/ldm_text2img/ldm_large_256_ddim.npy'
        )
    max_diff = np.abs(expected_image - image).max()
    assert max_diff < 0.001
