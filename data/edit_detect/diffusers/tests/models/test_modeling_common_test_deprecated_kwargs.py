def test_deprecated_kwargs(self):
    has_kwarg_in_model_class = 'kwargs' in inspect.signature(self.
        model_class.__init__).parameters
    has_deprecated_kwarg = len(self.model_class._deprecated_kwargs) > 0
    if has_kwarg_in_model_class and not has_deprecated_kwarg:
        raise ValueError(
            f'{self.model_class} has `**kwargs` in its __init__ method but has not defined any deprecated kwargs under the `_deprecated_kwargs` class attribute. Make sure to either remove `**kwargs` if there are no deprecated arguments or add the deprecated argument with `_deprecated_kwargs = [<deprecated_argument>]`'
            )
    if not has_kwarg_in_model_class and has_deprecated_kwarg:
        raise ValueError(
            f"{self.model_class} doesn't have `**kwargs` in its __init__ method but has defined deprecated kwargs under the `_deprecated_kwargs` class attribute. Make sure to either add the `**kwargs` argument to {self.model_class}.__init__ if there are deprecated arguments or remove the deprecated argument from `_deprecated_kwargs = [<deprecated_argument>]`"
            )
