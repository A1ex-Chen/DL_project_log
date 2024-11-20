@slow
def test_stable_diffusion_two_xl_mixture_of_denoiser(self):
    components = self.get_dummy_components()
    pipe_1 = StableDiffusionXLPipeline(**components).to(torch_device)
    pipe_1.unet.set_default_attn_processor()
    pipe_2 = StableDiffusionXLImg2ImgPipeline(**components).to(torch_device)
    pipe_2.unet.set_default_attn_processor()

    def assert_run_mixture(num_steps, split, scheduler_cls_orig,
        expected_tss, num_train_timesteps=pipe_1.scheduler.config.
        num_train_timesteps):
        inputs = self.get_dummy_inputs(torch_device)
        inputs['num_inference_steps'] = num_steps


        class scheduler_cls(scheduler_cls_orig):
            pass
        pipe_1.scheduler = scheduler_cls.from_config(pipe_1.scheduler.config)
        pipe_2.scheduler = scheduler_cls.from_config(pipe_2.scheduler.config)
        pipe_1.scheduler.set_timesteps(num_steps)
        expected_steps = pipe_1.scheduler.timesteps.tolist()
        if pipe_1.scheduler.order == 2:
            expected_steps_1 = list(filter(lambda ts: ts >= split,
                expected_tss))
            expected_steps_2 = expected_steps_1[-1:] + list(filter(lambda
                ts: ts < split, expected_tss))
            expected_steps = expected_steps_1 + expected_steps_2
        else:
            expected_steps_1 = list(filter(lambda ts: ts >= split,
                expected_tss))
            expected_steps_2 = list(filter(lambda ts: ts < split, expected_tss)
                )
        done_steps = []
        old_step = copy.copy(scheduler_cls.step)

        def new_step(self, *args, **kwargs):
            done_steps.append(args[1].cpu().item())
            return old_step(self, *args, **kwargs)
        scheduler_cls.step = new_step
        inputs_1 = {**inputs, **{'denoising_end': 1.0 - split /
            num_train_timesteps, 'output_type': 'latent'}}
        latents = pipe_1(**inputs_1).images[0]
        assert expected_steps_1 == done_steps, f'Failure with {scheduler_cls.__name__} and {num_steps} and {split}'
        inputs_2 = {**inputs, **{'denoising_start': 1.0 - split /
            num_train_timesteps, 'image': latents}}
        pipe_2(**inputs_2).images[0]
        assert expected_steps_2 == done_steps[len(expected_steps_1):]
        assert expected_steps == done_steps, f'Failure with {scheduler_cls.__name__} and {num_steps} and {split}'
    steps = 10
    for split in [300, 500, 700]:
        for scheduler_cls_timesteps in [(DDIMScheduler, [901, 801, 701, 601,
            501, 401, 301, 201, 101, 1]), (EulerDiscreteScheduler, [901, 
            801, 701, 601, 501, 401, 301, 201, 101, 1]), (
            DPMSolverMultistepScheduler, [901, 811, 721, 631, 541, 451, 361,
            271, 181, 91]), (UniPCMultistepScheduler, [901, 811, 721, 631, 
            541, 451, 361, 271, 181, 91]), (HeunDiscreteScheduler, [901.0, 
            801.0, 801.0, 701.0, 701.0, 601.0, 601.0, 501.0, 501.0, 401.0, 
            401.0, 301.0, 301.0, 201.0, 201.0, 101.0, 101.0, 1.0, 1.0])]:
            assert_run_mixture(steps, split, scheduler_cls_timesteps[0],
                scheduler_cls_timesteps[1])
    steps = 25
    for split in [300, 500, 700]:
        for scheduler_cls_timesteps in [(DDIMScheduler, [961, 921, 881, 841,
            801, 761, 721, 681, 641, 601, 561, 521, 481, 441, 401, 361, 321,
            281, 241, 201, 161, 121, 81, 41, 1]), (EulerDiscreteScheduler,
            [961.0, 921.0, 881.0, 841.0, 801.0, 761.0, 721.0, 681.0, 641.0,
            601.0, 561.0, 521.0, 481.0, 441.0, 401.0, 361.0, 321.0, 281.0, 
            241.0, 201.0, 161.0, 121.0, 81.0, 41.0, 1.0]), (
            DPMSolverMultistepScheduler, [951, 913, 875, 837, 799, 761, 723,
            685, 647, 609, 571, 533, 495, 457, 419, 381, 343, 305, 267, 229,
            191, 153, 115, 77, 39]), (UniPCMultistepScheduler, [951, 913, 
            875, 837, 799, 761, 723, 685, 647, 609, 571, 533, 495, 457, 419,
            381, 343, 305, 267, 229, 191, 153, 115, 77, 39]), (
            HeunDiscreteScheduler, [961.0, 921.0, 921.0, 881.0, 881.0, 
            841.0, 841.0, 801.0, 801.0, 761.0, 761.0, 721.0, 721.0, 681.0, 
            681.0, 641.0, 641.0, 601.0, 601.0, 561.0, 561.0, 521.0, 521.0, 
            481.0, 481.0, 441.0, 441.0, 401.0, 401.0, 361.0, 361.0, 321.0, 
            321.0, 281.0, 281.0, 241.0, 241.0, 201.0, 201.0, 161.0, 161.0, 
            121.0, 121.0, 81.0, 81.0, 41.0, 41.0, 1.0, 1.0])]:
            assert_run_mixture(steps, split, scheduler_cls_timesteps[0],
                scheduler_cls_timesteps[1])
