def test_inference_steps(self):
    for t, num_inference_steps in zip([99, 39, 39, 19], [10, 25, 26, 50]):
        self.check_over_forward(time_step=t, num_inference_steps=
            num_inference_steps)
