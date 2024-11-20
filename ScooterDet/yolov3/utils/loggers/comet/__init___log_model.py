def log_model(self, path, opt, epoch, fitness_score, best_model=False):
    if not self.save_model:
        return
    model_metadata = {'fitness_score': fitness_score[-1], 'epochs_trained':
        epoch + 1, 'save_period': opt.save_period, 'total_epochs': opt.epochs}
    model_files = glob.glob(f'{path}/*.pt')
    for model_path in model_files:
        name = Path(model_path).name
        self.experiment.log_model(self.model_name, file_or_folder=
            model_path, file_name=name, metadata=model_metadata, overwrite=True
            )
