@torch.no_grad()
@validate_hf_hub_args
def merge(self, pretrained_model_name_or_path_list: List[Union[str, os.
    PathLike]], **kwargs):
    """
        Returns a new pipeline object of the class 'DiffusionPipeline' with the merged checkpoints(weights) of the models passed
        in the argument 'pretrained_model_name_or_path_list' as a list.

        Parameters:
        -----------
            pretrained_model_name_or_path_list : A list of valid pretrained model names in the HuggingFace hub or paths to locally stored models in the HuggingFace format.

            **kwargs:
                Supports all the default DiffusionPipeline.get_config_dict kwargs viz..

                cache_dir, resume_download, force_download, proxies, local_files_only, token, revision, torch_dtype, device_map.

                alpha - The interpolation parameter. Ranges from 0 to 1.  It affects the ratio in which the checkpoints are merged. A 0.8 alpha
                    would mean that the first model checkpoints would affect the final result far less than an alpha of 0.2

                interp - The interpolation method to use for the merging. Supports "sigmoid", "inv_sigmoid", "add_diff" and None.
                    Passing None uses the default interpolation which is weighted sum interpolation. For merging three checkpoints, only "add_diff" is supported.

                force - Whether to ignore mismatch in model_config.json for the current models. Defaults to False.

                variant - which variant of a pretrained model to load, e.g. "fp16" (None)

        """
    cache_dir = kwargs.pop('cache_dir', None)
    resume_download = kwargs.pop('resume_download', False)
    force_download = kwargs.pop('force_download', False)
    proxies = kwargs.pop('proxies', None)
    local_files_only = kwargs.pop('local_files_only', False)
    token = kwargs.pop('token', None)
    variant = kwargs.pop('variant', None)
    revision = kwargs.pop('revision', None)
    torch_dtype = kwargs.pop('torch_dtype', None)
    device_map = kwargs.pop('device_map', None)
    alpha = kwargs.pop('alpha', 0.5)
    interp = kwargs.pop('interp', None)
    print('Received list', pretrained_model_name_or_path_list)
    print(f'Combining with alpha={alpha}, interpolation mode={interp}')
    checkpoint_count = len(pretrained_model_name_or_path_list)
    force = kwargs.pop('force', False)
    if checkpoint_count > 3 or checkpoint_count < 2:
        raise ValueError(
            'Received incorrect number of checkpoints to merge. Ensure that either 2 or 3 checkpoints are being passed.'
            )
    print('Received the right number of checkpoints')
    config_dicts = []
    for pretrained_model_name_or_path in pretrained_model_name_or_path_list:
        config_dict = DiffusionPipeline.load_config(
            pretrained_model_name_or_path, cache_dir=cache_dir,
            resume_download=resume_download, force_download=force_download,
            proxies=proxies, local_files_only=local_files_only, token=token,
            revision=revision)
        config_dicts.append(config_dict)
    comparison_result = True
    for idx in range(1, len(config_dicts)):
        comparison_result &= self._compare_model_configs(config_dicts[idx -
            1], config_dicts[idx])
        if not force and comparison_result is False:
            raise ValueError(
                'Incompatible checkpoints. Please check model_index.json for the models.'
                )
    print('Compatible model_index.json files found')
    cached_folders = []
    for pretrained_model_name_or_path, config_dict in zip(
        pretrained_model_name_or_path_list, config_dicts):
        folder_names = [k for k in config_dict.keys() if not k.startswith('_')]
        allow_patterns = [os.path.join(k, '*') for k in folder_names]
        allow_patterns += [WEIGHTS_NAME, SCHEDULER_CONFIG_NAME, CONFIG_NAME,
            ONNX_WEIGHTS_NAME, DiffusionPipeline.config_name]
        requested_pipeline_class = config_dict.get('_class_name')
        user_agent = {'diffusers': __version__, 'pipeline_class':
            requested_pipeline_class}
        cached_folder = pretrained_model_name_or_path if os.path.isdir(
            pretrained_model_name_or_path) else snapshot_download(
            pretrained_model_name_or_path, cache_dir=cache_dir,
            resume_download=resume_download, proxies=proxies,
            local_files_only=local_files_only, token=token, revision=
            revision, allow_patterns=allow_patterns, user_agent=user_agent)
        print('Cached Folder', cached_folder)
        cached_folders.append(cached_folder)
    final_pipe = DiffusionPipeline.from_pretrained(cached_folders[0],
        torch_dtype=torch_dtype, device_map=device_map, variant=variant)
    final_pipe.to(self.device)
    checkpoint_path_2 = None
    if len(cached_folders) > 2:
        checkpoint_path_2 = os.path.join(cached_folders[2])
    if interp == 'sigmoid':
        theta_func = CheckpointMergerPipeline.sigmoid
    elif interp == 'inv_sigmoid':
        theta_func = CheckpointMergerPipeline.inv_sigmoid
    elif interp == 'add_diff':
        theta_func = CheckpointMergerPipeline.add_difference
    else:
        theta_func = CheckpointMergerPipeline.weighted_sum
    for attr in final_pipe.config.keys():
        if not attr.startswith('_'):
            checkpoint_path_1 = os.path.join(cached_folders[1], attr)
            if os.path.exists(checkpoint_path_1):
                files = [*glob.glob(os.path.join(checkpoint_path_1,
                    '*.safetensors')), *glob.glob(os.path.join(
                    checkpoint_path_1, '*.bin'))]
                checkpoint_path_1 = files[0] if len(files) > 0 else None
            if len(cached_folders) < 3:
                checkpoint_path_2 = None
            else:
                checkpoint_path_2 = os.path.join(cached_folders[2], attr)
                if os.path.exists(checkpoint_path_2):
                    files = [*glob.glob(os.path.join(checkpoint_path_2,
                        '*.safetensors')), *glob.glob(os.path.join(
                        checkpoint_path_2, '*.bin'))]
                    checkpoint_path_2 = files[0] if len(files) > 0 else None
            if checkpoint_path_1 is None and checkpoint_path_2 is None:
                print(f'Skipping {attr}: not present in 2nd or 3d model')
                continue
            try:
                module = getattr(final_pipe, attr)
                if isinstance(module, bool):
                    continue
                theta_0 = getattr(module, 'state_dict')
                theta_0 = theta_0()
                update_theta_0 = getattr(module, 'load_state_dict')
                theta_1 = safetensors.torch.load_file(checkpoint_path_1
                    ) if checkpoint_path_1.endswith('.safetensors'
                    ) else torch.load(checkpoint_path_1, map_location='cpu')
                theta_2 = None
                if checkpoint_path_2:
                    theta_2 = safetensors.torch.load_file(checkpoint_path_2
                        ) if checkpoint_path_2.endswith('.safetensors'
                        ) else torch.load(checkpoint_path_2, map_location='cpu'
                        )
                if not theta_0.keys() == theta_1.keys():
                    print(f'Skipping {attr}: key mismatch')
                    continue
                if theta_2 and not theta_1.keys() == theta_2.keys():
                    print(f'Skipping {attr}:y mismatch')
            except Exception as e:
                print(f'Skipping {attr} do to an unexpected error: {str(e)}')
                continue
            print(f'MERGING {attr}')
            for key in theta_0.keys():
                if theta_2:
                    theta_0[key] = theta_func(theta_0[key], theta_1[key],
                        theta_2[key], alpha)
                else:
                    theta_0[key] = theta_func(theta_0[key], theta_1[key],
                        None, alpha)
            del theta_1
            del theta_2
            update_theta_0(theta_0)
            del theta_0
    return final_pipe
