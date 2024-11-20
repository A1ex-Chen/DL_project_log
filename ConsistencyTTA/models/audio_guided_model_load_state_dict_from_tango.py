def load_state_dict_from_tango(self, tango_state_dict: Mapping[str, Any],
    **kwargs):
    """ Load the teacher model from a pre-trained TANGO checkpoint and initialize
            the student models as well as its EMA copy with the teacher model weights.
        """
    new_state_dict = OrderedDict()
    modules = ['teacher', 'student', 'student_ema']
    for key, val in tango_state_dict.items():
        if 'unet' in key and '_unet' not in key:
            for module in modules:
                new_state_dict[f'{module}_{key}'] = val
        else:
            new_state_dict[key] = val
    try:
        return_info = self.load_state_dict(new_state_dict, strict=True)
    except:
        logger.info(
            'Strict loading failed. The loaded state_dict may not match the target model.'
            )
        return_info = self.load_state_dict(new_state_dict, strict=False)
        print(f'Keys that are not loaded: {return_info.missing_keys}')
        assert len(return_info.unexpected_keys
            ) == 0, f'Redundant keys in state_dict: {return_info.unexpected_keys}'
    if self.use_lora:
        self.setup_lora()
    self.student_ema_unet = deepcopy(self.student_unet)
    self.student_ema_unet.requires_grad_(False)
    return return_info
