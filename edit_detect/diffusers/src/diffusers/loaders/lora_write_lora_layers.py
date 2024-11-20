@staticmethod
def write_lora_layers(state_dict: Dict[str, torch.Tensor], save_directory:
    str, is_main_process: bool, weight_name: str, save_function: Callable,
    safe_serialization: bool):
    if os.path.isfile(save_directory):
        logger.error(
            f'Provided path ({save_directory}) should be a directory, not a file'
            )
        return
    if save_function is None:
        if safe_serialization:

            def save_function(weights, filename):
                return safetensors.torch.save_file(weights, filename,
                    metadata={'format': 'pt'})
        else:
            save_function = torch.save
    os.makedirs(save_directory, exist_ok=True)
    if weight_name is None:
        if safe_serialization:
            weight_name = LORA_WEIGHT_NAME_SAFE
        else:
            weight_name = LORA_WEIGHT_NAME
    save_path = Path(save_directory, weight_name).as_posix()
    save_function(state_dict, save_path)
    logger.info(f'Model weights saved in {save_path}')
