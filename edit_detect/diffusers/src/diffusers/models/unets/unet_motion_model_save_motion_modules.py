def save_motion_modules(self, save_directory: str, is_main_process: bool=
    True, safe_serialization: bool=True, variant: Optional[str]=None,
    push_to_hub: bool=False, **kwargs) ->None:
    state_dict = self.state_dict()
    motion_state_dict = {}
    for k, v in state_dict.items():
        if 'motion_modules' in k:
            motion_state_dict[k] = v
    adapter = MotionAdapter(block_out_channels=self.config[
        'block_out_channels'], motion_layers_per_block=self.config[
        'layers_per_block'], motion_norm_num_groups=self.config[
        'norm_num_groups'], motion_num_attention_heads=self.config[
        'motion_num_attention_heads'], motion_max_seq_length=self.config[
        'motion_max_seq_length'], use_motion_mid_block=self.config[
        'use_motion_mid_block'])
    adapter.load_state_dict(motion_state_dict)
    adapter.save_pretrained(save_directory=save_directory, is_main_process=
        is_main_process, safe_serialization=safe_serialization, variant=
        variant, push_to_hub=push_to_hub, **kwargs)
