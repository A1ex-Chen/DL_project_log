def on_pretrain_routine_end(self, paths):
    if self.opt.resume:
        return
    for path in paths:
        self.log_asset(str(path))
    if self.upload_dataset and not self.resume:
        self.upload_dataset_artifact()
    return
