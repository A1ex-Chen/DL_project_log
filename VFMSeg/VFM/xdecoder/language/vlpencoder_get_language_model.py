@register_model
def get_language_model(cfg, **kwargs):
    return LanguageEncoder(cfg)
