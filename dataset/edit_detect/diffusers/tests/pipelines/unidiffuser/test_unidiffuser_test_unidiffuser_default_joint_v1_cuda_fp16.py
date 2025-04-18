@require_torch_gpu
def test_unidiffuser_default_joint_v1_cuda_fp16(self):
    device = 'cuda'
    unidiffuser_pipe = UniDiffuserPipeline.from_pretrained(
        'hf-internal-testing/unidiffuser-test-v1', torch_dtype=torch.float16)
    unidiffuser_pipe = unidiffuser_pipe.to(device)
    unidiffuser_pipe.set_progress_bar_config(disable=None)
    unidiffuser_pipe.set_joint_mode()
    assert unidiffuser_pipe.mode == 'joint'
    inputs = self.get_dummy_inputs_with_latents(device)
    del inputs['prompt']
    del inputs['image']
    inputs['data_type'] = 1
    sample = unidiffuser_pipe(**inputs)
    image = sample.images
    text = sample.text
    assert image.shape == (1, 32, 32, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_img_slice = np.array([0.5049, 0.5498, 0.5854, 0.3052, 0.446, 
        0.6489, 0.5122, 0.481, 0.6138])
    assert np.abs(image_slice.flatten() - expected_img_slice).max() < 0.001
    expected_text_prefix = '" This This'
    assert text[0][:len(expected_text_prefix)] == expected_text_prefix
