def load_ip_adapter_face_id(self, pretrained_model_name_or_path_or_dict,
    weight_name, **kwargs):
    cache_dir = kwargs.pop('cache_dir', None)
    force_download = kwargs.pop('force_download', False)
    resume_download = kwargs.pop('resume_download', False)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', None)
    token = kwargs.pop('token', None)
    revision = kwargs.pop('revision', None)
    subfolder = kwargs.pop('subfolder', None)
    user_agent = {'file_type': 'attn_procs_weights', 'framework': 'pytorch'}
    model_file = _get_model_file(pretrained_model_name_or_path_or_dict,
        weights_name=weight_name, cache_dir=cache_dir, force_download=
        force_download, resume_download=resume_download, proxies=proxies,
        local_files_only=local_files_only, token=token, revision=revision,
        subfolder=subfolder, user_agent=user_agent)
    if weight_name.endswith('.safetensors'):
        state_dict = {'image_proj': {}, 'ip_adapter': {}}
        with safe_open(model_file, framework='pt', device='cpu') as f:
            for key in f.keys():
                if key.startswith('image_proj.'):
                    state_dict['image_proj'][key.replace('image_proj.', '')
                        ] = f.get_tensor(key)
                elif key.startswith('ip_adapter.'):
                    state_dict['ip_adapter'][key.replace('ip_adapter.', '')
                        ] = f.get_tensor(key)
    else:
        state_dict = torch.load(model_file, map_location='cpu')
    self._load_ip_adapter_weights(state_dict)
