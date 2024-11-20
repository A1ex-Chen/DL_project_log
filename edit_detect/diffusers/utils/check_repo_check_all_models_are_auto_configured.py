def check_all_models_are_auto_configured():
    """Check all models are each in an auto class."""
    missing_backends = []
    if not is_torch_available():
        missing_backends.append('PyTorch')
    if not is_flax_available():
        missing_backends.append('Flax')
    if len(missing_backends) > 0:
        missing = ', '.join(missing_backends)
        if os.getenv('TRANSFORMERS_IS_CI', '').upper() in ENV_VARS_TRUE_VALUES:
            raise Exception(
                f'Full quality checks require all backends to be installed (with `pip install -e .[dev]` in the Transformers repo, the following are missing: {missing}.'
                )
        else:
            warnings.warn(
                f"Full quality checks require all backends to be installed (with `pip install -e .[dev]` in the Transformers repo, the following are missing: {missing}. While it's probably fine as long as you didn't make any change in one of those backends modeling files, you should probably execute the command above to be on the safe side."
                )
    modules = get_model_modules()
    all_auto_models = get_all_auto_configured_models()
    failures = []
    for module in modules:
        new_failures = check_models_are_auto_configured(module, all_auto_models
            )
        if new_failures is not None:
            failures += new_failures
    if len(failures) > 0:
        raise Exception(f'There were {len(failures)} failures:\n' + '\n'.
            join(failures))
