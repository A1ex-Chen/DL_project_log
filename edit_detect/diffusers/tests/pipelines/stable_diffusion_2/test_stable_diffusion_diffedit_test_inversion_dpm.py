def test_inversion_dpm(self):
    device = 'cpu'
    components = self.get_dummy_components()
    scheduler_args = {'beta_start': 0.00085, 'beta_end': 0.012,
        'beta_schedule': 'scaled_linear'}
    components['scheduler'] = DPMSolverMultistepScheduler(**scheduler_args)
    components['inverse_scheduler'] = DPMSolverMultistepInverseScheduler(**
        scheduler_args)
    pipe = self.pipeline_class(**components)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inversion_inputs(device)
    image = pipe.invert(**inputs).images
    image_slice = image[0, -1, -3:, -3:]
    self.assertEqual(image.shape, (2, 32, 32, 3))
    expected_slice = np.array([0.5305, 0.4673, 0.5314, 0.5308, 0.4886, 
        0.5279, 0.5142, 0.4724, 0.4892])
    max_diff = np.abs(image_slice.flatten() - expected_slice).max()
    self.assertLessEqual(max_diff, 0.001)
