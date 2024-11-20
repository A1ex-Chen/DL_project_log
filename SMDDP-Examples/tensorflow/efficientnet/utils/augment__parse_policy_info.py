def _parse_policy_info(name: Text, prob: float, level: float, replace_value:
    List[int], cutout_const: float, translate_const: float) ->Tuple[Any,
    float, Any]:
    """Return the function that corresponds to `name` and update `level` param."""
    func = NAME_TO_FUNC[name]
    args = level_to_arg(cutout_const, translate_const)[name](level)
    if name in REPLACE_FUNCS:
        args = tuple(list(args) + [replace_value])
    return func, prob, args
