def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.6345, 0.5395, 0.5611, 0.5403, 
            0.583, 0.5855, 0.5193, 0.5443, 0.5211])
    return super().test_ip_adapter_single(from_simple=True,
        expected_pipe_slice=expected_pipe_slice)
