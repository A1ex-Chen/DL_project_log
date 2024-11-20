def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.5813, 0.61, 0.4756, 0.5057, 0.572,
            0.4632, 0.5177, 0.5125, 0.4718])
    return super().test_ip_adapter_single(from_multi=True,
        expected_pipe_slice=expected_pipe_slice)
