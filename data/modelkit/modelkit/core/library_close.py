def close(self):
    for model in self.models.values():
        if isinstance(model, Model):
            model.close()
        if isinstance(model, AsyncModel):
            AsyncToSync(model.close)()
