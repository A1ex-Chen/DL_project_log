def on_train_end(self, last, best, plots, epoch, results):
    if plots:
        plot_results(file=self.save_dir / 'results.csv')
    files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for
        x in ('F1', 'PR', 'P', 'R'))]
    files = [(self.save_dir / f) for f in files if (self.save_dir / f).exists()
        ]
    self.logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")
    if self.tb and not self.clearml:
        for f in files:
            self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch,
                dataformats='HWC')
    if self.wandb:
        self.wandb.log(dict(zip(self.keys[3:10], results)))
        self.wandb.log({'Results': [wandb.Image(str(f), caption=f.name) for
            f in files]})
        if not self.opt.evolve:
            wandb.log_artifact(str(best if best.exists() else last), type=
                'model', name=f'run_{self.wandb.wandb_run.id}_model',
                aliases=['latest', 'best', 'stripped'])
        self.wandb.finish_run()
    if self.clearml:
        if not self.opt.evolve:
            self.clearml.task.update_output_model(model_path=str(best if
                best.exists() else last), name='Best Model')
