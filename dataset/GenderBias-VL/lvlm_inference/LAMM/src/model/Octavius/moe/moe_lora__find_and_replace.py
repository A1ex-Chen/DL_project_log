def _find_and_replace(self, adapter_name):
    lora_config = self.peft_config[adapter_name]
    loaded_in_8bit = getattr(self.model, 'is_loaded_in_8bit', False)
    if loaded_in_8bit:
        raise NotImplementedError
    is_target_modules_in_base_model = False
    kwargs = {'r': lora_config.r, 'num_experts': lora_config.num_experts,
        'gate_mode': lora_config.gate_mode, 'lora_alpha': lora_config.
        lora_alpha, 'lora_dropout': lora_config.lora_dropout,
        'fan_in_fan_out': lora_config.fan_in_fan_out, 'init_lora_weights':
        lora_config.init_lora_weights}
    key_list = [key for key, _ in self.model.named_modules()]
    for key in key_list:
        if isinstance(lora_config.target_modules, str):
            target_module_found = re.fullmatch(lora_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for
                target_key in lora_config.target_modules)
        if target_module_found:
            if not is_target_modules_in_base_model:
                is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            if hasattr(target, 'bias'):
                bias = target.bias is not None
            else:
                bias = False
            if isinstance(target, MoeLoraLayer) and isinstance(target, nn.
                Linear):
                target.update_moe_layer(adapter_name, lora_config.r,
                    lora_config.num_experts, lora_config.lora_alpha,
                    lora_config.lora_dropout, lora_config.init_lora_weights)
            elif isinstance(target, nn.Linear):
                in_features, out_features = (target.in_features, target.
                    out_features)
                if kwargs['fan_in_fan_out']:
                    warnings.warn(
                        'fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. Setting fan_in_fan_out to False.'
                        )
                    kwargs['fan_in_fan_out'
                        ] = lora_config.fan_in_fan_out = False
                new_module = MoeLinear(adapter_name, in_features,
                    out_features, bias=bias, **kwargs)
            elif isinstance(target, nn.Conv1D):
                in_features, out_features = target.weight.ds_shape if hasattr(
                    target.weight, 'ds_shape') else target.weight.shape
                if not kwargs['fan_in_fan_out']:
                    warnings.warn(
                        'fan_in_fan_out is set to False but the target module is `torch.nn.Conv1D`. Setting fan_in_fan_out to True.'
                        )
                    kwargs['fan_in_fan_out'
                        ] = lora_config.fan_in_fan_out = True
                new_module = MoeLinear(adapter_name, in_features,
                    out_features, bias=bias, **kwargs)
            else:
                raise RuntimeError(
                    f'Target module {target} is not supported. Currently, only ``torch.nn.Linear`, torch.nn.Conv1D` and `torch.nn.Embedding` are supported.'
                    )
            self._replace_module(parent, target_name, new_module, target)
    if not is_target_modules_in_base_model:
        raise ValueError(
            f'Target modules {lora_config.target_modules} not found in the base model. Please check the target modules and try again.'
            )
