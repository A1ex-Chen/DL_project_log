def test_inference_steps(self):
    for t, num_inference_steps in zip([1, 10, 50], [10, 50, 500]):
        self.check_over_forward(time_step=t, num_inference_steps=
            num_inference_steps)
