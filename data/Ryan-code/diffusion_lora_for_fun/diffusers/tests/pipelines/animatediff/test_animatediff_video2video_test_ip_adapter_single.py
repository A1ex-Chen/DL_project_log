def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.4947, 0.478, 0.434, 0.4666, 
            0.4028, 0.4645, 0.4915, 0.4101, 0.4308, 0.4581, 0.3582, 0.4953,
            0.4466, 0.5348, 0.5863, 0.5299, 0.5213, 0.5017])
    return super().test_ip_adapter_single(expected_pipe_slice=
        expected_pipe_slice)
