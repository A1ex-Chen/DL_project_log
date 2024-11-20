def check_model_type_doc_match():
    """Check all doc pages have a corresponding model type."""
    model_doc_folder = Path(PATH_TO_DOC) / 'model_doc'
    model_docs = [m.stem for m in model_doc_folder.glob('*.md')]
    model_types = list(diffusers.models.auto.configuration_auto.
        MODEL_NAMES_MAPPING.keys())
    model_types = [(MODEL_TYPE_TO_DOC_MAPPING[m] if m in
        MODEL_TYPE_TO_DOC_MAPPING else m) for m in model_types]
    errors = []
    for m in model_docs:
        if m not in model_types and m != 'auto':
            close_matches = get_close_matches(m, model_types)
            error_message = f'{m} is not a proper model identifier.'
            if len(close_matches) > 0:
                close_matches = '/'.join(close_matches)
                error_message += f' Did you mean {close_matches}?'
            errors.append(error_message)
    if len(errors) > 0:
        raise ValueError(
            'Some model doc pages do not match any existing model type:\n' +
            '\n'.join(errors) +
            """
You can add any missing model type to the `MODEL_NAMES_MAPPING` constant in models/auto/configuration_auto.py."""
            )
