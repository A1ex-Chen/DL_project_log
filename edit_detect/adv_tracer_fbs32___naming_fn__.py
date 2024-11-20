def __naming_fn__(self):
    return (
        f"res-{self.ds_root.replace('images/', '')}-{self.model_id}-{self.loss_metric}-lr{self.lr}-bs{self.batch_size}"
        )
