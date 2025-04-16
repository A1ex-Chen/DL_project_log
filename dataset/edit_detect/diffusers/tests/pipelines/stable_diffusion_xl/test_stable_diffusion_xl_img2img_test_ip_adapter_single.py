def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.5174, 0.4512, 0.5006, 0.6273, 
            0.516, 0.6825, 0.6655, 0.584, 0.5675])
    return super().test_ip_adapter_single(expected_pipe_slice=
        expected_pipe_slice)
