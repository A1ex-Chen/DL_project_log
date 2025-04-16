@require_torch_gpu
def test_unidiffuser_default_img2text_v1_cuda_fp16(self):
    device = 'cuda'
    unidiffuser_pipe = UniDiffuserPipeline.from_pretrained(
        'hf-internal-testing/unidiffuser-test-v1', torch_dtype=torch.float16)
    unidiffuser_pipe = unidiffuser_pipe.to(device)
    unidiffuser_pipe.set_progress_bar_config(disable=None)
    unidiffuser_pipe.set_image_to_text_mode()
    assert unidiffuser_pipe.mode == 'img2text'
    inputs = self.get_dummy_inputs_with_latents(device)
    del inputs['prompt']
    inputs['data_type'] = 1
    text = unidiffuser_pipe(**inputs).text
    expected_text_prefix = '" This This'
    assert text[0][:len(expected_text_prefix)] == expected_text_prefix
