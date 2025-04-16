def prepare_interactive(model_args, preprocessor: Dict[str, Any]):
    conv_args = model_args.conv_args
    tokenize_kwargs = conv_args.get('tokenize_kwargs', {})
    conv_template = conv_args.get('conv_template', 'vicuna_v1.1')
    conv_template = partial(get_conv_template, name=conv_template)
    transforms = conv_args.get('transforms', None)
    if transforms is not None:
        transforms = TRANSFORMS.build(transforms)
    process_func = {}
    for k, v in model_args.process_func_args.items():
        process_func[k] = FUNCTIONS.build(cfg=v)
    ds = SingleImageInteractive(preprocessor=preprocessor, process_func=
        process_func, tokenize_kwargs=tokenize_kwargs, conv_template=
        conv_template, training_args=None, transforms=transforms, mode='test')
    return ds
