def check_over_forward(self, time_step=0, **forward_kwargs):
    kwargs = dict(self.forward_default_kwargs)
    num_inference_steps = kwargs.pop('num_inference_steps', None)
    sample, _ = self.dummy_sample
    residual = 0.1 * sample
    dummy_past_residuals = jnp.array([residual + 0.2, residual + 0.15, 
        residual + 0.1, residual + 0.05])
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.create_state()
        state = scheduler.set_timesteps(state, num_inference_steps, shape=
            sample.shape)
        scheduler.ets = dummy_past_residuals[:]
        with tempfile.TemporaryDirectory() as tmpdirname:
            scheduler.save_config(tmpdirname)
            new_scheduler, new_state = scheduler_class.from_pretrained(
                tmpdirname)
            new_state = new_scheduler.set_timesteps(new_state,
                num_inference_steps, shape=sample.shape)
            new_state.replace(ets=dummy_past_residuals[:])
        output, state = scheduler.step_prk(state, residual, time_step,
            sample, **kwargs)
        new_output, new_state = new_scheduler.step_prk(new_state, residual,
            time_step, sample, **kwargs)
        assert jnp.sum(jnp.abs(output - new_output)
            ) < 1e-05, 'Scheduler outputs are not identical'
        output, _ = scheduler.step_plms(state, residual, time_step, sample,
            **kwargs)
        new_output, _ = new_scheduler.step_plms(new_state, residual,
            time_step, sample, **kwargs)
        assert jnp.sum(jnp.abs(output - new_output)
            ) < 1e-05, 'Scheduler outputs are not identical'
