def modelcard_has_connected_pipeline(self, model_id):
    modelcard = ModelCard.load(model_id)
    connected_pipes = {prefix: getattr(modelcard.data, prefix, [None])[0] for
        prefix in CONNECTED_PIPES_KEYS}
    connected_pipes = {k: v for k, v in connected_pipes.items() if v is not
        None}
    return len(connected_pipes) > 0
