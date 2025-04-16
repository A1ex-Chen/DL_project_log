def check_models_are_auto_configured(module, all_auto_models):
    """Check models defined in module are each in an auto class."""
    defined_models = get_models(module)
    failures = []
    for model_name, _ in defined_models:
        if model_name not in all_auto_models and not ignore_unautoclassed(
            model_name):
            failures.append(
                f'{model_name} is defined in {module.__name__} but is not present in any of the auto mapping. If that is intended behavior, add its name to `IGNORE_NON_AUTO_CONFIGURED` in the file `utils/check_repo.py`.'
                )
    return failures
