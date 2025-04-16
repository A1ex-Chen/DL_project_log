def not_overriden(func: Callable) ->CallableWithAttribute:
    func_with_attributes = cast(CallableWithAttribute, func)
    func_with_attributes.__not_overriden__ = True
    return func_with_attributes
