def load_state_dict_from_tango(self, tango_state_dict: Mapping[str, Any],
    stage1_state_dict: Mapping[str, Any]=None):
    """ Load the teacher diffusion model from a pre-trained TANGO checkpoint;
            initialize the student model and its EMA copies with the teacher weights.
        """
    new_state_dict = OrderedDict()
    student_modules = ['student', 'student_target', 'student_ema']
    for key, val in tango_state_dict.items():
        if 'unet' in key and '_unet' not in key:
            new_state_dict[f'teacher_{key}'] = val
            if stage1_state_dict is None:
                for module in student_modules:
                    new_state_dict[f'{module}_{key}'] = val
        else:
            new_state_dict[key] = val
    if stage1_state_dict is not None:
        for key, val in stage1_state_dict.items():
            if 'student_ema' in key:
                aft_key = key.split('student_ema_')[-1]
                for module in student_modules:
                    new_state_dict[f'{module}_{aft_key}'] = val
    try:
        return_info = self.load_state_dict(new_state_dict, strict=True)
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
    if self.use_lora:
        self.setup_lora()
    self.student_target_unet.requires_grad_(False)
    self.student_ema_unet.requires_grad_(False)
    self.teacher_unet.requires_grad_(False)
    return return_info
