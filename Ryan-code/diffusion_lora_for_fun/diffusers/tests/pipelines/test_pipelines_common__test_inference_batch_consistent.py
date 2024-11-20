def _test_inference_batch_consistent(self, batch_sizes=[2],
    additional_params_copy_to_batched_inputs=['num_inference_steps'],
    batch_generator=True):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['generator'] = self.get_generator(0)
    logger = logging.get_logger(pipe.__module__)
    logger.setLevel(level=diffusers.logging.FATAL)
    batched_inputs = []
    for batch_size in batch_sizes:
        batched_input = {}
        batched_input.update(inputs)
        for name in self.batch_params:
            if name not in inputs:
                continue
            value = inputs[name]
            if name == 'prompt':
                len_prompt = len(value)
                batched_input[name] = [value[:len_prompt // i] for i in
                    range(1, batch_size + 1)]
                batched_input[name][-1] = 100 * 'very long'
            else:
                batched_input[name] = batch_size * [value]
        if batch_generator and 'generator' in inputs:
            batched_input['generator'] = [self.get_generator(i) for i in
                range(batch_size)]
        if 'batch_size' in inputs:
            batched_input['batch_size'] = batch_size
        batched_inputs.append(batched_input)
    logger.setLevel(level=diffusers.logging.WARNING)
    for batch_size, batched_input in zip(batch_sizes, batched_inputs):
        output = pipe(**batched_input)
        assert len(output[0]) == batch_size
