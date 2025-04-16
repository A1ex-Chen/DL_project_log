def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.4003, 0.3718, 0.2863, 0.55, 
            0.5587, 0.3772, 0.4617, 0.4961, 0.4417])
    return super().test_ip_adapter_single(expected_pipe_slice=
        expected_pipe_slice)
