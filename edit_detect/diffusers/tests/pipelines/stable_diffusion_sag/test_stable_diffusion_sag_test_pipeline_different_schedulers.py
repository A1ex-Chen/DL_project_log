def test_pipeline_different_schedulers(self):
    pipeline = self.pipeline_class(**self.get_dummy_components())
    inputs = self.get_dummy_inputs('cpu')
    expected_image_size = 16, 16, 3
    for scheduler_cls in [DDIMScheduler, DEISMultistepScheduler,
        DPMSolverMultistepScheduler]:
        pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.
            config)
        image = pipeline(**inputs).images[0]
        shape = image.shape
        assert shape == expected_image_size
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.
        scheduler.config)
    with self.assertRaises(ValueError):
        image = pipeline(**inputs).images[0]
