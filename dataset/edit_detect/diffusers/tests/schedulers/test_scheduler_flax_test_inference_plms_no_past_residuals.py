def test_inference_plms_no_past_residuals(self):
    with self.assertRaises(ValueError):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.create_state()
        scheduler.step_plms(state, self.dummy_sample, 1, self.dummy_sample
            ).prev_sample
