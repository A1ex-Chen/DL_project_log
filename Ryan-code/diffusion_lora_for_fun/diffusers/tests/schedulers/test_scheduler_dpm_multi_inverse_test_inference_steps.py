def test_inference_steps(self):
    for num_inference_steps in [1, 2, 3, 5, 10, 50, 100, 999, 1000]:
        self.check_over_forward(num_inference_steps=num_inference_steps,
            time_step=0)
