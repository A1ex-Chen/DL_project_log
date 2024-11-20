def set_adapters(self, adapter_names: Union[List[str], str],
    adapter_weights: Optional[Union[float, Dict, List[float], List[Dict]]]=None
    ):
    adapter_names = [adapter_names] if isinstance(adapter_names, str
        ) else adapter_names
    adapter_weights = copy.deepcopy(adapter_weights)
    if not isinstance(adapter_weights, list):
        adapter_weights = [adapter_weights] * len(adapter_names)
    if len(adapter_names) != len(adapter_weights):
        raise ValueError(
            f'Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(adapter_weights)}'
            )
    (unet_lora_weights, text_encoder_lora_weights, text_encoder_2_lora_weights
        ) = [], [], []
    list_adapters = self.get_list_adapters()
    all_adapters = {adapter for adapters in list_adapters.values() for
        adapter in adapters}
    invert_list_adapters = {adapter: [part for part, adapters in
        list_adapters.items() if adapter in adapters] for adapter in
        all_adapters}
    for adapter_name, weights in zip(adapter_names, adapter_weights):
        if isinstance(weights, dict):
            unet_lora_weight = weights.pop('unet', None)
            text_encoder_lora_weight = weights.pop('text_encoder', None)
            text_encoder_2_lora_weight = weights.pop('text_encoder_2', None)
            if len(weights) > 0:
                raise ValueError(
                    f"Got invalid key '{weights.keys()}' in lora weight dict for adapter {adapter_name}."
                    )
            if text_encoder_2_lora_weight is not None and not hasattr(self,
                'text_encoder_2'):
                logger.warning(
                    'Lora weight dict contains text_encoder_2 weights but will be ignored because pipeline does not have text_encoder_2.'
                    )
            for part_weight, part_name in zip([unet_lora_weight,
                text_encoder_lora_weight, text_encoder_2_lora_weight], [
                'unet', 'text_encoder', 'text_encoder_2']):
                if (part_weight is not None and part_name not in
                    invert_list_adapters[adapter_name]):
                    logger.warning(
                        f"Lora weight dict for adapter '{adapter_name}' contains {part_name}, but this will be ignored because {adapter_name} does not contain weights for {part_name}. Valid parts for {adapter_name} are: {invert_list_adapters[adapter_name]}."
                        )
        else:
            unet_lora_weight = weights
            text_encoder_lora_weight = weights
            text_encoder_2_lora_weight = weights
        unet_lora_weights.append(unet_lora_weight)
        text_encoder_lora_weights.append(text_encoder_lora_weight)
        text_encoder_2_lora_weights.append(text_encoder_2_lora_weight)
    unet = getattr(self, self.unet_name) if not hasattr(self, 'unet'
        ) else self.unet
    unet.set_adapters(adapter_names, unet_lora_weights)
    if hasattr(self, 'text_encoder'):
        self.set_adapters_for_text_encoder(adapter_names, self.text_encoder,
            text_encoder_lora_weights)
    if hasattr(self, 'text_encoder_2'):
        self.set_adapters_for_text_encoder(adapter_names, self.
            text_encoder_2, text_encoder_2_lora_weights)
