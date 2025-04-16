def check_device_map_is_respected(self, model, device_map):
    for param_name, param in model.named_parameters():
        while len(param_name) > 0 and param_name not in device_map:
            param_name = '.'.join(param_name.split('.')[:-1])
        if param_name not in device_map:
            raise ValueError(
                'device map is incomplete, it does not contain any device for `param_name`.'
                )
        param_device = device_map[param_name]
        if param_device in ['cpu', 'disk']:
            self.assertEqual(param.device, torch.device('meta'))
        else:
            self.assertEqual(param.device, torch.device(param_device))
