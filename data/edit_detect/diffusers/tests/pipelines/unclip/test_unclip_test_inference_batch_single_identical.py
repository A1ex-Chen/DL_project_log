@skip_mps
def test_inference_batch_single_identical(self):
    additional_params_copy_to_batched_inputs = ['prior_num_inference_steps',
        'decoder_num_inference_steps', 'super_res_num_inference_steps']
    self._test_inference_batch_single_identical(
        additional_params_copy_to_batched_inputs=
        additional_params_copy_to_batched_inputs, expected_max_diff=0.005)
