def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.6265, 0.5441, 0.5384, 0.5446, 
            0.581, 0.5908, 0.5414, 0.5428, 0.5353])
    return super().test_ip_adapter_single(expected_pipe_slice=
        expected_pipe_slice)
