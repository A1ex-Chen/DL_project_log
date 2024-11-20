def create_model(model_name: str, pretrained: str='', jit: bool=False,
    force_quick_gelu: bool=False, pretrained_image: bool=False):
    model_name = model_name.replace('/', '-')
    if pretrained and pretrained.lower() == 'openai':
        raise NotImplementedError
    else:
        if model_name in _MODEL_CONFIGS:
            logger.info(f'Loading {model_name} model config.')
            model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
        else:
            logging.error(
                f'Model config for {model_name} not found; available models {list_models()}.'
                )
            raise RuntimeError(f'Model config for {model_name} not found.')
        if force_quick_gelu:
            model_cfg['quick_gelu'] = True
        if pretrained_image:
            if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'
        model = ClipVisualOnly(**model_cfg)
        if not pretrained:
            dim = model.visual.transformer.resblocks[0
                ].attn.in_proj_weight.shape[0] // 3
            for resblock in model.visual.transformer.resblocks:
                resblock.ts_attn.q_proj.weight = nn.Parameter(resblock.attn
                    .in_proj_weight[:dim].clone())
                resblock.ts_attn.q_proj.bias = nn.Parameter(resblock.attn.
                    in_proj_bias[:dim].clone())
                resblock.ts_attn.k_proj.weight = nn.Parameter(resblock.attn
                    .in_proj_weight[dim:2 * dim].clone())
                resblock.ts_attn.k_proj.bias = nn.Parameter(resblock.attn.
                    in_proj_bias[dim:2 * dim].clone())
                resblock.ts_attn.v_proj.weight = nn.Parameter(resblock.attn
                    .in_proj_weight[2 * dim:].clone())
                resblock.ts_attn.v_proj.bias = nn.Parameter(resblock.attn.
                    in_proj_bias[2 * dim:].clone())
                resblock.ts_attn.out_proj.weight = nn.Parameter(resblock.
                    attn.out_proj.weight.clone())
                resblock.ts_attn.out_proj.bias = nn.Parameter(resblock.attn
                    .out_proj.bias.clone())
                resblock.attn = None
        if pretrained:
            logger.info(f'Loading {model_name} checkpoint from: {pretrained}')
            checkpoint_path = ''
            url = get_pretrained_url(model_name, pretrained)
            if url:
                checkpoint_path = download_pretrained(url)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained
            if checkpoint_path:
                logging.info(
                    f'Loading pretrained {model_name} weights ({pretrained}).')
                load_checkpoint4vision_only(model, checkpoint_path, strict=
                    False)
                dim = model.visual.transformer.resblocks[0
                    ].attn.in_proj_weight.shape[0] // 3
                for resblock in model.visual.transformer.resblocks:
                    resblock.ts_attn.q_proj.weight = nn.Parameter(resblock.
                        attn.in_proj_weight[:dim].clone())
                    resblock.ts_attn.q_proj.bias = nn.Parameter(resblock.
                        attn.in_proj_bias[:dim].clone())
                    resblock.ts_attn.k_proj.weight = nn.Parameter(resblock.
                        attn.in_proj_weight[dim:2 * dim].clone())
                    resblock.ts_attn.k_proj.bias = nn.Parameter(resblock.
                        attn.in_proj_bias[dim:2 * dim].clone())
                    resblock.ts_attn.v_proj.weight = nn.Parameter(resblock.
                        attn.in_proj_weight[2 * dim:].clone())
                    resblock.ts_attn.v_proj.bias = nn.Parameter(resblock.
                        attn.in_proj_bias[2 * dim:].clone())
                    resblock.ts_attn.out_proj.weight = nn.Parameter(resblock
                        .attn.out_proj.weight.clone())
                    resblock.ts_attn.out_proj.bias = nn.Parameter(resblock.
                        attn.out_proj.bias.clone())
                    resblock.attn = None
            else:
                logging.warning(
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                    )
                raise RuntimeError(
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                    )
        if jit:
            model = torch.jit.script(model)
    return model
