@classmethod
def save_lora_weights(self, save_directory: Union[str, os.PathLike],
    unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]]=None,
    text_encoder_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor
    ]]=None, text_encoder_2_lora_layers: Dict[str, Union[torch.nn.Module,
    torch.Tensor]]=None, is_main_process: bool=True, weight_name: str=None,
    save_function: Callable=None, safe_serialization: bool=True):
    state_dict = {}

    def pack_weights(layers, prefix):
        layers_weights = layers.state_dict() if isinstance(layers, torch.nn
            .Module) else layers
        layers_state_dict = {f'{prefix}.{module_name}': param for 
            module_name, param in layers_weights.items()}
        return layers_state_dict
    if not (unet_lora_layers or text_encoder_lora_layers or
        text_encoder_2_lora_layers):
        raise ValueError(
            'You must pass at least one of `unet_lora_layers`, `text_encoder_lora_layers` or `text_encoder_2_lora_layers`.'
            )
    if unet_lora_layers:
        state_dict.update(pack_weights(unet_lora_layers, 'unet'))
    if text_encoder_lora_layers and text_encoder_2_lora_layers:
        state_dict.update(pack_weights(text_encoder_lora_layers,
            'text_encoder'))
        state_dict.update(pack_weights(text_encoder_2_lora_layers,
            'text_encoder_2'))
    self.write_lora_layers(state_dict=state_dict, save_directory=
        save_directory, is_main_process=is_main_process, weight_name=
        weight_name, save_function=save_function, safe_serialization=
        safe_serialization)
