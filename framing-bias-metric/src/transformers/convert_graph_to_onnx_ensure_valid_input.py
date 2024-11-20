def ensure_valid_input(model, tokens, input_names):
    """
    Ensure input are presented in the correct order, without any Non

    Args:
        model: The model used to forward the input data
        tokens: BatchEncoding holding the input data
        input_names: The name of the inputs

    Returns: Tuple

    """
    print('Ensuring inputs are in correct order')
    model_args_name = model.forward.__code__.co_varnames
    model_args, ordered_input_names = [], []
    for arg_name in model_args_name[1:]:
        if arg_name in input_names:
            ordered_input_names.append(arg_name)
            model_args.append(tokens[arg_name])
        else:
            print(f'{arg_name} is not present in the generated input list.')
            break
    print('Generated inputs order: {}'.format(ordered_input_names))
    return ordered_input_names, tuple(model_args)
