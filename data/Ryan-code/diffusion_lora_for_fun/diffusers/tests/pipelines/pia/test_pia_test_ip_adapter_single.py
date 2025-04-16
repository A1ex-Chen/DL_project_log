def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.5609, 0.5756, 0.483, 0.442, 
            0.4547, 0.5129, 0.3779, 0.4042, 0.3772, 0.445, 0.571, 0.5536, 
            0.4835, 0.4308, 0.5578, 0.5578, 0.4395, 0.544, 0.6051, 0.4651, 
            0.6258, 0.5662, 0.3988, 0.5108, 0.4153, 0.3993, 0.4803])
    return super().test_ip_adapter_single(expected_pipe_slice=
        expected_pipe_slice)
