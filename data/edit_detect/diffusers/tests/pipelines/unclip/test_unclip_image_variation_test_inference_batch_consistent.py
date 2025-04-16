def test_inference_batch_consistent(self):
    additional_params_copy_to_batched_inputs = ['decoder_num_inference_steps',
        'super_res_num_inference_steps']
    if torch_device == 'mps':
        batch_sizes = [2, 3]
        self._test_inference_batch_consistent(batch_sizes=batch_sizes,
            additional_params_copy_to_batched_inputs=
            additional_params_copy_to_batched_inputs)
    else:
        self._test_inference_batch_consistent(
            additional_params_copy_to_batched_inputs=
            additional_params_copy_to_batched_inputs)
