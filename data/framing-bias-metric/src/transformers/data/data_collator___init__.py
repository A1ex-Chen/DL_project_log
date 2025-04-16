def __init__(self, *args, **kwargs):
    warnings.warn(
        'DataCollatorForSOP is deprecated and will be removed in a future version, you can now use DataCollatorForLanguageModeling instead.'
        , FutureWarning)
