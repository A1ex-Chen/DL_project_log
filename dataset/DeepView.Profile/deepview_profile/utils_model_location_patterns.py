def model_location_patterns():
    return ['./transformers/models[/\\w+/]+\\w+.py',
        './transformers/integrations[/\\w+/]+\\w+.py',
        './diffusers/models[/\\w+/]+\\w+.py']
