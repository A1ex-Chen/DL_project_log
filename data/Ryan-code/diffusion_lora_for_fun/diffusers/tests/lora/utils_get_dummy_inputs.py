def get_dummy_inputs(self, with_generator=True):
    batch_size = 1
    sequence_length = 10
    num_channels = 4
    sizes = 32, 32
    generator = torch.manual_seed(0)
    noise = floats_tensor((batch_size, num_channels) + sizes)
    input_ids = torch.randint(1, sequence_length, size=(batch_size,
        sequence_length), generator=generator)
    pipeline_inputs = {'prompt': 'A painting of a squirrel eating a burger',
        'num_inference_steps': 5, 'guidance_scale': 6.0, 'output_type': 'np'}
    if with_generator:
        pipeline_inputs.update({'generator': generator})
    return noise, input_ids, pipeline_inputs
