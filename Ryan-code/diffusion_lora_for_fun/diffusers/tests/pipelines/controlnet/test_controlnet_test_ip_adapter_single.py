def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.5264, 0.3203, 0.1602, 0.8235, 
            0.6332, 0.4593, 0.7226, 0.7777, 0.478])
    return super().test_ip_adapter_single(expected_pipe_slice=
        expected_pipe_slice)
