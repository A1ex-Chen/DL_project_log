def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.5293, 0.7339, 0.6642, 0.395, 
            0.5212, 0.5175, 0.7002, 0.5907, 0.5182])
    return super().test_ip_adapter_single(expected_pipe_slice=
        expected_pipe_slice)
