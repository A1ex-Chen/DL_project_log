def test_free_init(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.set_progress_bar_config(disable=None)
    pipe.to(torch_device)
    inputs_normal = self.get_dummy_inputs(torch_device)
    frames_normal = pipe(**inputs_normal).frames[0]
    pipe.enable_free_init(num_iters=2, use_fast_sampling=True, method=
        'butterworth', order=4, spatial_stop_frequency=0.25,
        temporal_stop_frequency=0.25)
    inputs_enable_free_init = self.get_dummy_inputs(torch_device)
    frames_enable_free_init = pipe(**inputs_enable_free_init).frames[0]
    pipe.disable_free_init()
    inputs_disable_free_init = self.get_dummy_inputs(torch_device)
    frames_disable_free_init = pipe(**inputs_disable_free_init).frames[0]
    sum_enabled = np.abs(to_np(frames_normal) - to_np(frames_enable_free_init)
        ).sum()
    max_diff_disabled = np.abs(to_np(frames_normal) - to_np(
        frames_disable_free_init)).max()
    self.assertGreater(sum_enabled, 10.0,
        'Enabling of FreeInit should lead to results different from the default pipeline results'
        )
    self.assertLess(max_diff_disabled, 0.0001,
        'Disabling of FreeInit should lead to results similar to the default pipeline results'
        )
