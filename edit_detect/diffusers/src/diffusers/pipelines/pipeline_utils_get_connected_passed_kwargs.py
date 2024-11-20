def get_connected_passed_kwargs(prefix):
    connected_passed_class_obj = {k.replace(f'{prefix}_', ''): w for k, w in
        passed_class_obj.items() if k.split('_')[0] == prefix}
    connected_passed_pipe_kwargs = {k.replace(f'{prefix}_', ''): w for k, w in
        passed_pipe_kwargs.items() if k.split('_')[0] == prefix}
    connected_passed_kwargs = {**connected_passed_class_obj, **
        connected_passed_pipe_kwargs}
    return connected_passed_kwargs
