def load_pretrained(model_args, training_args) ->Tuple[nn.Module, PREPROCESSOR
    ]:
    type_ = model_args.type
    if type_ == 'shikra':
        return load_pretrained_shikra(model_args, training_args)
    else:
        assert False
