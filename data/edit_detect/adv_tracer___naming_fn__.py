def __naming_fn__(self):
    return (
        f"res-{self.ds_root.replace('images/', '')}-{self.model_id}-{self.loss_metric}-lr{self.lr}-{self.loss_metric}-it{self.max_iter}-lrnc{self.num_cycles}-bs{self.batch_size}"
        )
