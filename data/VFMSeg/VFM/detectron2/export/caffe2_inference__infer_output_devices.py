def _infer_output_devices(self, inputs):
    """
        Returns:
            list[str]: list of device for each external output
        """

    def _get_device_type(torch_tensor):
        assert torch_tensor.device.type in ['cpu', 'cuda']
        assert torch_tensor.device.index == 0
        return torch_tensor.device.type
    predict_net = self.net.Proto()
    input_device_types = {(name, 0): _get_device_type(tensor) for name,
        tensor in zip(self._input_blobs, inputs)}
    device_type_map = infer_device_type(predict_net, known_status=
        input_device_types, device_name_style='pytorch')
    ssa, versions = core.get_ssa(predict_net)
    versioned_outputs = [(name, versions[name]) for name in predict_net.
        external_output]
    output_devices = [device_type_map[outp] for outp in versioned_outputs]
    return output_devices
