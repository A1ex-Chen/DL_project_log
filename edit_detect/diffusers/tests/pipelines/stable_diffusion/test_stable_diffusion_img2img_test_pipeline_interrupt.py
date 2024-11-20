def test_pipeline_interrupt(self):
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionImg2ImgPipeline(**components)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    prompt = 'hey'
    num_inference_steps = 3


    class PipelineState:

        def __init__(self):
            self.state = []

        def apply(self, pipe, i, t, callback_kwargs):
            self.state.append(callback_kwargs['latents'])
            return callback_kwargs
    pipe_state = PipelineState()
    sd_pipe(prompt, image=inputs['image'], num_inference_steps=
        num_inference_steps, output_type='np', generator=torch.Generator(
        'cpu').manual_seed(0), callback_on_step_end=pipe_state.apply).images
    interrupt_step_idx = 1

    def callback_on_step_end(pipe, i, t, callback_kwargs):
        if i == interrupt_step_idx:
            pipe._interrupt = True
        return callback_kwargs
    output_interrupted = sd_pipe(prompt, image=inputs['image'],
        num_inference_steps=num_inference_steps, output_type='latent',
        generator=torch.Generator('cpu').manual_seed(0),
        callback_on_step_end=callback_on_step_end).images
    intermediate_latent = pipe_state.state[interrupt_step_idx]
    assert torch.allclose(intermediate_latent, output_interrupted, atol=0.0001)
