def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.5552, 0.5569, 0.4725, 0.4348, 
            0.4994, 0.4632, 0.5142, 0.5012, 0.47])
    return super().test_ip_adapter_single(expected_pipe_slice=
        expected_pipe_slice)
