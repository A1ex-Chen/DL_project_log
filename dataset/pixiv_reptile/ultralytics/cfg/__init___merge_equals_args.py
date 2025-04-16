def merge_equals_args(args: List[str]) ->List[str]:
    """
    Merges arguments around isolated '=' args in a list of strings. The function considers cases where the first
    argument ends with '=' or the second starts with '=', as well as when the middle one is an equals sign.

    Args:
        args (List[str]): A list of strings where each element is an argument.

    Returns:
        (List[str]): A list of strings where the arguments around isolated '=' are merged.

    Example:
        The function modifies the argument list as follows:
        ```python
        args = ["arg1", "=", "value"]
        new_args = merge_equals_args(args)
        print(new_args)  # Output: ["arg1=value"]

        args = ["arg1=", "value"]
        new_args = merge_equals_args(args)
        print(new_args)  # Output: ["arg1=value"]

        args = ["arg1", "=value"]
        new_args = merge_equals_args(args)
        print(new_args)  # Output: ["arg1=value"]
        ```
    """
    new_args = []
    for i, arg in enumerate(args):
        if arg == '=' and 0 < i < len(args) - 1:
            new_args[-1] += f'={args[i + 1]}'
            del args[i + 1]
        elif arg.endswith('=') and i < len(args) - 1 and '=' not in args[i + 1
            ]:
            new_args.append(f'{arg}{args[i + 1]}')
            del args[i + 1]
        elif arg.startswith('=') and i > 0:
            new_args[-1] += arg
        else:
            new_args.append(arg)
    return new_args
