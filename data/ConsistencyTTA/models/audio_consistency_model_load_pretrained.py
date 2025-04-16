def load_pretrained(self, state_dict: Mapping[str, Any], strict: bool=True):
    """ This function converts parameter names before loading the state_dict,
            so that we can use the model trained via older implementations.
        """
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if 'consistency_unet' in key:
            aft_key = key.split('consistency_unet')[-1]
            new_state_dict[f'student_unet{aft_key}'] = val
        elif 'consistency_ema_' in key:
            aft_key = key.split('consistency_ema_')[-1]
            new_state_dict[f'student_target_{aft_key}'] = val
            if f'student_ema_{aft_key}' not in new_state_dict.keys():
                new_state_dict[f'student_ema_{aft_key}'] = val
        elif 'consistency_slow_ema_' in key:
            aft_key = key.split('consistency_slow_ema_')[-1]
            new_state_dict[f'student_ema_{aft_key}'] = val
        elif 'diffusion_unet' in key:
            aft_key = key.split('diffusion_unet')[-1]
            new_state_dict[f'teacher_unet{aft_key}'] = val
        elif 'loss.' in key and 'vae.' not in key:
            aft_key = key.split('loss.')[-1]
            new_state_dict[f'stft_loss.{aft_key}'] = val
        elif 'vae.' not in key:
            new_state_dict[key] = val
    try:
        return self.load_state_dict(new_state_dict, strict=True)
    except:
        logger.info(
            "Strict loading failed. The loaded state_dict may not match the target model. This is okay if 'Keys that are not loaded' is an empty list."
            )
        return_info = self.load_state_dict(new_state_dict, strict=False)
        missing_keys = [key for key in return_info.missing_keys if 'vae' not in
            key and 'loss.' not in key]
        redundant_keys = [key for key in return_info.unexpected_keys if 
            'vae' not in key and 'loss.' not in key]
        print(f'Keys that are not loaded: {missing_keys}')
        assert len(redundant_keys
            ) == 0, f'Redundant keys in state_dict: {return_info.unexpected_keys}'
        return return_info
