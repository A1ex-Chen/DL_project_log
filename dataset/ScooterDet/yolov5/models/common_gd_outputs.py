def gd_outputs(gd):
    name_list, input_list = [], []
    for node in gd.node:
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if
        not x.startswith('NoOp'))
