def load_model_dict_into_meta(model, state_dict: OrderedDict, device:
    Optional[Union[str, torch.device]]=None, dtype: Optional[Union[str,
    torch.dtype]]=None, model_name_or_path: Optional[str]=None) ->List[str]:
    device = device or torch.device('cpu')
    dtype = dtype or torch.float32
    accepts_dtype = 'dtype' in set(inspect.signature(
        set_module_tensor_to_device).parameters.keys())
    unexpected_keys = []
    empty_state_dict = model.state_dict()
    for param_name, param in state_dict.items():
        if param_name not in empty_state_dict:
            unexpected_keys.append(param_name)
            continue
        if empty_state_dict[param_name].shape != param.shape:
            model_name_or_path_str = (f'{model_name_or_path} ' if 
                model_name_or_path is not None else '')
            raise ValueError(
                f'Cannot load {model_name_or_path_str}because {param_name} expected shape {empty_state_dict[param_name]}, but got {param.shape}. If you want to instead overwrite randomly initialized weights, please make sure to pass both `low_cpu_mem_usage=False` and `ignore_mismatched_sizes=True`. For more information, see also: https://github.com/huggingface/diffusers/issues/1619#issuecomment-1345604389 as an example.'
                )
        if accepts_dtype:
            set_module_tensor_to_device(model, param_name, device, value=
                param, dtype=dtype)
        else:
            set_module_tensor_to_device(model, param_name, device, value=param)
    return unexpected_keys
