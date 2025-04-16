def test_inference_batch_single_identical(self, batch_size=2,
    expected_max_diff=0.0001, additional_params_copy_to_batched_inputs=[
    'num_inference_steps']):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    for components in pipe.components.values():
        if hasattr(components, 'set_default_attn_processor'):
            components.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['generator'] = self.get_generator(0)
    logger = logging.get_logger(pipe.__module__)
    logger.setLevel(level=diffusers.logging.FATAL)
    batched_inputs = {}
    batched_inputs.update(inputs)
    for name in self.batch_params:
        if name not in inputs:
            continue
        value = inputs[name]
        if name == 'prompt':
            len_prompt = len(value)
            batched_inputs[name] = [value[:len_prompt // i] for i in range(
                1, batch_size + 1)]
            batched_inputs[name][-1] = 100 * 'very long'
        else:
            batched_inputs[name] = batch_size * [value]
    if 'generator' in inputs:
        batched_inputs['generator'] = [self.get_generator(i) for i in range
            (batch_size)]
    if 'batch_size' in inputs:
        batched_inputs['batch_size'] = batch_size
    for arg in additional_params_copy_to_batched_inputs:
        batched_inputs[arg] = inputs[arg]
    output = pipe(**inputs)
    output_batch = pipe(**batched_inputs)
    assert output_batch[0].shape[0] == batch_size
    max_diff = np.abs(to_np(output_batch[0][0]) - to_np(output[0][0])).max()
    assert max_diff < expected_max_diff
