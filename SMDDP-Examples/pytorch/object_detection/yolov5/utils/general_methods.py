def methods(instance):
    return [f for f in dir(instance) if callable(getattr(instance, f)) and 
        not f.startswith('__')]
