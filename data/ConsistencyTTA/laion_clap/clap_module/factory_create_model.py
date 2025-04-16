def create_model(amodel_name: str, tmodel_name: str, pretrained: str='',
    precision: str='fp32', device: torch.device=torch.device('cpu'), jit:
    bool=False, force_quick_gelu: bool=False, openai_model_cache_dir: str=
    os.path.expanduser('~/.cache/clip'), skip_params=True, pretrained_audio:
    str='', pretrained_text: str='', enable_fusion: bool=False, fusion_type:
    str='None'):
    amodel_name = amodel_name.replace('/', '-')
    pretrained_orig = pretrained
    pretrained = pretrained.lower()
    if pretrained == 'openai':
        if amodel_name in _MODEL_CONFIGS:
            logging.info(f'Loading {amodel_name} model config.')
            model_cfg = deepcopy(_MODEL_CONFIGS[amodel_name])
        else:
            logging.error(
                f'Model config for {amodel_name} not found; available models {list_models()}.'
                )
            raise RuntimeError(f'Model config for {amodel_name} not found.')
        logging.info(f'Loading pretrained ViT-B-16 text encoder from OpenAI.')
        model_cfg['text_cfg']['model_type'] = tmodel_name
        model = load_openai_model('ViT-B-16', model_cfg, device=device, jit
            =jit, cache_dir=openai_model_cache_dir, enable_fusion=
            enable_fusion, fusion_type=fusion_type)
        if precision == 'amp' or precision == 'fp32':
            model = model.float()
    else:
        if amodel_name in _MODEL_CONFIGS:
            logging.info(f'Loading {amodel_name} model config.')
            model_cfg = deepcopy(_MODEL_CONFIGS[amodel_name])
        else:
            logging.error(
                f'Model config for {amodel_name} not found; available models {list_models()}.'
                )
            raise RuntimeError(f'Model config for {amodel_name} not found.')
        if force_quick_gelu:
            model_cfg['quick_gelu'] = True
        model_cfg['text_cfg']['model_type'] = tmodel_name
        model_cfg['enable_fusion'] = enable_fusion
        model_cfg['fusion_type'] = fusion_type
        model = CLAP(**model_cfg)
        if pretrained:
            checkpoint_path = ''
            url = get_pretrained_url(amodel_name, pretrained)
            if url:
                checkpoint_path = download_pretrained(url, root=
                    openai_model_cache_dir)
            elif os.path.exists(pretrained_orig):
                checkpoint_path = pretrained_orig
            if checkpoint_path:
                logging.info(
                    f'Loading pretrained {amodel_name}-{tmodel_name} weights ({pretrained}).'
                    )
                ckpt = load_state_dict(checkpoint_path, skip_params=True)
                model.load_state_dict(ckpt)
                param_names = [n for n, p in model.named_parameters()]
                for n in param_names:
                    print(n, '\t', 'Loaded' if n in ckpt else 'Unloaded')
            else:
                logging.warning(
                    f'Pretrained weights ({pretrained}) not found for model {amodel_name}.'
                    )
                raise RuntimeError(
                    f'Pretrained weights ({pretrained}) not found for model {amodel_name}.'
                    )
        if pretrained_audio:
            if amodel_name.startswith('PANN'):
                if 'Cnn14_mAP' in pretrained_audio:
                    audio_ckpt = torch.load(pretrained_audio, map_location=
                        'cpu')
                    audio_ckpt = audio_ckpt['model']
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if ('spectrogram_extractor' not in key and 
                            'logmel_extractor' not in key):
                            v = audio_ckpt.pop(key)
                            audio_ckpt['audio_branch.' + key] = v
                elif os.path.basename(pretrained_audio).startswith('PANN'):
                    audio_ckpt = torch.load(pretrained_audio, map_location=
                        'cpu')
                    audio_ckpt = audio_ckpt['state_dict']
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if key.startswith('sed_model'):
                            v = audio_ckpt.pop(key)
                            audio_ckpt['audio_branch.' + key[10:]] = v
                elif os.path.basename(pretrained_audio).startswith('finetuned'
                    ):
                    audio_ckpt = torch.load(pretrained_audio, map_location=
                        'cpu')
                else:
                    raise ValueError('Unknown audio checkpoint')
            elif amodel_name.startswith('HTSAT'):
                if 'HTSAT_AudioSet_Saved' in pretrained_audio:
                    audio_ckpt = torch.load(pretrained_audio, map_location=
                        'cpu')
                    audio_ckpt = audio_ckpt['state_dict']
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if key.startswith('sed_model') and (
                            'spectrogram_extractor' not in key and 
                            'logmel_extractor' not in key):
                            v = audio_ckpt.pop(key)
                            audio_ckpt['audio_branch.' + key[10:]] = v
                elif os.path.basename(pretrained_audio).startswith('HTSAT'):
                    audio_ckpt = torch.load(pretrained_audio, map_location=
                        'cpu')
                    audio_ckpt = audio_ckpt['state_dict']
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if key.startswith('sed_model'):
                            v = audio_ckpt.pop(key)
                            audio_ckpt['audio_branch.' + key[10:]] = v
                elif os.path.basename(pretrained_audio).startswith('finetuned'
                    ):
                    audio_ckpt = torch.load(pretrained_audio, map_location=
                        'cpu')
                else:
                    raise ValueError('Unknown audio checkpoint')
            else:
                raise f'this audio encoder pretrained checkpoint is not support'
            model.load_state_dict(audio_ckpt, strict=False)
            logging.info(
                f'Loading pretrained {amodel_name} weights ({pretrained_audio}).'
                )
            param_names = [n for n, p in model.named_parameters()]
            for n in param_names:
                print(n, '\t', 'Loaded' if n in audio_ckpt else 'Unloaded')
        model.to(device=device)
        if precision == 'fp16':
            assert device.type != 'cpu'
            convert_weights_to_fp16(model)
        if jit:
            model = torch.jit.script(model)
    return model, model_cfg
