def filter_torch_scope_torch12(node):
    """
    Extracts and formats the module name from a PyTorch graph node's scope name.
    """
    scope = node.scopeName().replace('Flatten/', '', 1).replace('Flatten',
        '', 1)
    scope_list = re.findall('\\[.*?\\]', scope)
    module_name = ''
    if len(scope_list) >= 2:
        module_name = '.'.join(token.strip('[]') for token in scope_list[1:])
    return module_name
