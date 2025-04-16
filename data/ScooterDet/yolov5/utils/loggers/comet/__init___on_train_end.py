def on_train_end(self, files, save_dir, last, best, epoch, results):
    if self.comet_log_predictions:
        curr_epoch = self.experiment.curr_epoch
        self.experiment.log_asset_data(self.metadata_dict,
            'image-metadata.json', epoch=curr_epoch)
    for f in files:
        self.log_asset(f, metadata={'epoch': epoch})
    self.log_asset(f'{save_dir}/results.csv', metadata={'epoch': epoch})
    if not self.opt.evolve:
        model_path = str(best if best.exists() else last)
        name = Path(model_path).name
        if self.save_model:
            self.experiment.log_model(self.model_name, file_or_folder=
                model_path, file_name=name, overwrite=True)
    if hasattr(self.opt, 'comet_optimizer_id'):
        metric = results.get(self.opt.comet_optimizer_metric)
        self.experiment.log_other('optimizer_metric_value', metric)
    self.finish_run()
