def test_inference_batch_consistent(self, batch_sizes=[2, 4, 13],
    additional_params_copy_to_batched_inputs=['num_inference_steps']):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    logger = logging.get_logger(pipe.__module__)
    logger.setLevel(level=diffusers.logging.FATAL)
    for batch_size in batch_sizes:
        batched_inputs = {}
        for name, value in inputs.items():
            if name in self.batch_params:
                if name == 'prompt':
                    len_prompt = len(value)
                    batched_inputs[name] = [value[:len_prompt // i] for i in
                        range(1, batch_size + 1)]
                    batched_inputs[name][-1] = 100 * 'very long'
                elif name == 'image':
                    batched_images = []
                    for image in value:
                        batched_images.append(batch_size * [image])
                    batched_inputs[name] = batched_images
                else:
                    batched_inputs[name] = batch_size * [value]
            elif name == 'batch_size':
                batched_inputs[name] = batch_size
            else:
                batched_inputs[name] = value
        for arg in additional_params_copy_to_batched_inputs:
            batched_inputs[arg] = inputs[arg]
        batched_inputs['output_type'] = 'np'
        if self.pipeline_class.__name__ == 'DanceDiffusionPipeline':
            batched_inputs.pop('output_type')
        output = pipe(**batched_inputs)
        assert len(output[0]) == batch_size
        batched_inputs['output_type'] = 'np'
        if self.pipeline_class.__name__ == 'DanceDiffusionPipeline':
            batched_inputs.pop('output_type')
        output = pipe(**batched_inputs)[0]
        assert output.shape[0] == batch_size
    logger.setLevel(level=diffusers.logging.WARNING)
