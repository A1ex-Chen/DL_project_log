def test_deprecated_kwargs(self):
    for scheduler_class in self.scheduler_classes:
        has_kwarg_in_model_class = 'kwargs' in inspect.signature(
            scheduler_class.__init__).parameters
        has_deprecated_kwarg = len(scheduler_class._deprecated_kwargs) > 0
        if has_kwarg_in_model_class and not has_deprecated_kwarg:
            raise ValueError(
                f'{scheduler_class} has `**kwargs` in its __init__ method but has not defined any deprecated kwargs under the `_deprecated_kwargs` class attribute. Make sure to either remove `**kwargs` if there are no deprecated arguments or add the deprecated argument with `_deprecated_kwargs = [<deprecated_argument>]`'
                )
        if not has_kwarg_in_model_class and has_deprecated_kwarg:
            raise ValueError(
                f"{scheduler_class} doesn't have `**kwargs` in its __init__ method but has defined deprecated kwargs under the `_deprecated_kwargs` class attribute. Make sure to either add the `**kwargs` argument to {self.model_class}.__init__ if there are deprecated arguments or remove the deprecated argument from `_deprecated_kwargs = [<deprecated_argument>]`"
                )
