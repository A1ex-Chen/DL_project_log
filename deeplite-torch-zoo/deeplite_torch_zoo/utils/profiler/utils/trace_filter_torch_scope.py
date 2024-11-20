def filter_torch_scope(node):
    """
    Extracts and formats the module name from a torch.Node traced scope name
    Relies on torchscript-based onnx trace: torch.onnx.trace

    Input: torch.Node node
    Output: str scope

    The node scope collected by torch.onnx.trace is string containing a
    '/' separated list of ModuleType::module_name
    The purpose of this function is to extract the module_names from this list
    Named submodules will appear in the list as expected, i.e. Conv2d[conv1]
    Unnamed submodules in a Sequential module or ModuleList will not always appear as expected
    This is a known issue https://github.com/pytorch/pytorch/issues/90439
    if the Sequential module's forward function is called:
        node.scope = Model::/Sequential::layers/Conv2d::layers.0
        Sequential module is present in scope list, so remove repeated 'layers' name
    if the submodule is called directly: (for m in sequential: m(x))
        node.scope = Model::/Conv2d::layers.0
        sequential module is not present in scope list, so preserve 'layers' name

    Future work:
    switch to torch.fx or torch dynamo tracing since torchscript tracing is not actively supported
    """
    if not TORCH_2_0:
        return filter_torch_scope_torch12(node)
    scope = node.scopeName()
    if scope == '':
        return scope
    module_pairs = scope.split('/')
    if len(module_pairs) == 1:
        return module_pairs[0]
    module_names = []
    module_types = []
    for module_pair in module_pairs:
        type_name = module_pair.split('::')
        if len(type_name) == 1:
            continue
        module_type, module_name = type_name[0], type_name[-1]
        module_name_split = module_name.split('.')
        if len(module_name_split) == 1:
            module_names.append(module_name)
        else:
            names_to_add = [module_name_split[-1]]
            types_to_add = [module_type]
            module_names_idx = len(module_names) - 1
            split_idx = len(module_name_split) - 2
            last_saved_name = module_names[module_names_idx]
            for split_idx in range(len(module_name_split) - 2, -1, -1):
                parent_name = module_name_split[split_idx]
                if parent_name == last_saved_name:
                    break
                names_to_add.append(parent_name)
                types_to_add.append('__hidden_sequential__')
            module_names.extend(names_to_add[::-1])
            module_types.extend(types_to_add[::-1])
    filtered_scope = '.'.join(module_names[1:])
    return filtered_scope
