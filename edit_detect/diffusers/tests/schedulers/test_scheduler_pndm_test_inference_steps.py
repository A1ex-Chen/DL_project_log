def test_inference_steps(self):
    for t, num_inference_steps in zip([1, 5, 10], [10, 50, 100]):
        self.check_over_forward(num_inference_steps=num_inference_steps)
