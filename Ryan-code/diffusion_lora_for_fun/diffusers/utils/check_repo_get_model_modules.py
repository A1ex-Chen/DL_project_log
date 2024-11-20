def get_model_modules():
    """Get the model modules inside the transformers library."""
    _ignore_modules = ['modeling_auto', 'modeling_encoder_decoder',
        'modeling_marian', 'modeling_mmbt', 'modeling_outputs',
        'modeling_retribert', 'modeling_utils', 'modeling_flax_auto',
        'modeling_flax_encoder_decoder', 'modeling_flax_utils',
        'modeling_speech_encoder_decoder',
        'modeling_flax_speech_encoder_decoder',
        'modeling_flax_vision_encoder_decoder',
        'modeling_transfo_xl_utilities', 'modeling_tf_auto',
        'modeling_tf_encoder_decoder', 'modeling_tf_outputs',
        'modeling_tf_pytorch_utils', 'modeling_tf_utils',
        'modeling_tf_transfo_xl_utilities',
        'modeling_tf_vision_encoder_decoder', 'modeling_vision_encoder_decoder'
        ]
    modules = []
    for model in dir(diffusers.models):
        if not model.startswith('__'):
            model_module = getattr(diffusers.models, model)
            for submodule in dir(model_module):
                if submodule.startswith('modeling'
                    ) and submodule not in _ignore_modules:
                    modeling_module = getattr(model_module, submodule)
                    if inspect.ismodule(modeling_module):
                        modules.append(modeling_module)
    return modules
