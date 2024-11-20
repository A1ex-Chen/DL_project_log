@classmethod
def from_pretrained(cls, path, model_cls) ->'EMAModel':
    _, ema_kwargs = model_cls.load_config(path, return_unused_kwargs=True)
    model = model_cls.from_pretrained(path)
    ema_model = cls(model.parameters(), model_cls=model_cls, model_config=
        model.config)
    ema_model.load_state_dict(ema_kwargs)
    return ema_model
