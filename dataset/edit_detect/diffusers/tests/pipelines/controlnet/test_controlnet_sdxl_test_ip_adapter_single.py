def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.6832, 0.5703, 0.546, 0.63, 0.5856,
            0.6034, 0.4494, 0.4613, 0.5036])
    return super().test_ip_adapter_single(from_ssd1b=True,
        expected_pipe_slice=expected_pipe_slice)
