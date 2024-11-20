def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.4932, 0.5092, 0.5135, 0.5517, 
            0.5626, 0.6621, 0.649, 0.5021, 0.5441])
    return super().test_ip_adapter_single(expected_pipe_slice=
        expected_pipe_slice)
