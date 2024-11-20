def _unwrap_model(model):
    """Unwraps a model."""
    if is_compiled_module(model):
        model = model._orig_mod
    if is_peft_available():
        from peft import PeftModel
        if isinstance(model, PeftModel):
            model = model.base_model.model
    return model
