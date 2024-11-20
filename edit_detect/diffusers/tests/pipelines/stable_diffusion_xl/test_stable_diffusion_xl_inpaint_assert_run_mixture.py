def assert_run_mixture(num_steps, split_1, split_2, scheduler_cls_orig,
    num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps):
    inputs = self.get_dummy_inputs(torch_device)
    inputs['num_inference_steps'] = num_steps


    class scheduler_cls(scheduler_cls_orig):
        pass
    pipe_1.scheduler = scheduler_cls.from_config(pipe_1.scheduler.config)
    pipe_2.scheduler = scheduler_cls.from_config(pipe_2.scheduler.config)
    pipe_3.scheduler = scheduler_cls.from_config(pipe_3.scheduler.config)
    pipe_1.scheduler.set_timesteps(num_steps)
    expected_steps = pipe_1.scheduler.timesteps.tolist()
    split_1_ts = num_train_timesteps - int(round(num_train_timesteps * split_1)
        )
    split_2_ts = num_train_timesteps - int(round(num_train_timesteps * split_2)
        )
    if pipe_1.scheduler.order == 2:
        expected_steps_1 = list(filter(lambda ts: ts >= split_1_ts,
            expected_steps))
        expected_steps_2 = expected_steps_1[-1:] + list(filter(lambda ts: 
            ts >= split_2_ts and ts < split_1_ts, expected_steps))
        expected_steps_3 = expected_steps_2[-1:] + list(filter(lambda ts: 
            ts < split_2_ts, expected_steps))
        expected_steps = expected_steps_1 + expected_steps_2 + expected_steps_3
    else:
        expected_steps_1 = list(filter(lambda ts: ts >= split_1_ts,
            expected_steps))
        expected_steps_2 = list(filter(lambda ts: ts >= split_2_ts and ts <
            split_1_ts, expected_steps))
        expected_steps_3 = list(filter(lambda ts: ts < split_2_ts,
            expected_steps))
    done_steps = []
    old_step = copy.copy(scheduler_cls.step)

    def new_step(self, *args, **kwargs):
        done_steps.append(args[1].cpu().item())
        return old_step(self, *args, **kwargs)
    scheduler_cls.step = new_step
    inputs_1 = {**inputs, **{'denoising_end': split_1, 'output_type': 'latent'}
        }
    latents = pipe_1(**inputs_1).images[0]
    assert expected_steps_1 == done_steps, f'Failure with {scheduler_cls.__name__} and {num_steps} and {split_1} and {split_2}'
    inputs_2 = {**inputs, **{'denoising_start': split_1, 'denoising_end':
        split_2, 'image': latents, 'output_type': 'latent'}}
    pipe_2(**inputs_2).images[0]
    assert expected_steps_2 == done_steps[len(expected_steps_1):]
    inputs_3 = {**inputs, **{'denoising_start': split_2, 'image': latents}}
    pipe_3(**inputs_3).images[0]
    assert expected_steps_3 == done_steps[len(expected_steps_1) + len(
        expected_steps_2):]
    assert expected_steps == done_steps, f'Failure with {scheduler_cls.__name__} and {num_steps} and {split_1} and {split_2}'
