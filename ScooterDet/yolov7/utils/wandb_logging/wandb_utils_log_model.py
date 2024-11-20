def log_model(self, path, opt, epoch, fitness_score, best_model=False):
    model_artifact = wandb.Artifact('run_' + wandb.run.id + '_model', type=
        'model', metadata={'original_url': str(path), 'epochs_trained': 
        epoch + 1, 'save period': opt.save_period, 'project': opt.project,
        'total_epochs': opt.epochs, 'fitness_score': fitness_score})
    model_artifact.add_file(str(path / 'last.pt'), name='last.pt')
    wandb.log_artifact(model_artifact, aliases=['latest', 'epoch ' + str(
        self.current_epoch), 'best' if best_model else ''])
    print('Saving model artifact on epoch ', epoch + 1)
