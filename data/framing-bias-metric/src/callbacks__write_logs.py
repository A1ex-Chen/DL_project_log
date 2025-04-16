@rank_zero_only
def _write_logs(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
    type_path: str, save_generations=True) ->None:
    logger.info(
        f'***** {type_path} results at step {trainer.global_step:05d} *****')
    metrics = trainer.callback_metrics
    trainer.logger.log_metrics({k: v for k, v in metrics.items() if k not in
        ['log', 'progress_bar', 'preds']})
    od = Path(pl_module.hparams.output_dir)
    if type_path == 'test':
        suffix = pl_module.hparams.custom_pred_file_suffix
        results_file = od / f'test_results_{suffix}.txt'
        generations_file = od / f'test_generations_{suffix}.txt'
    else:
        results_file = (od /
            f'{type_path}_results/{trainer.global_step:05d}.txt')
        generations_file = (od /
            f'{type_path}_generations/{trainer.global_step:05d}.txt')
        results_file.parent.mkdir(exist_ok=True)
        generations_file.parent.mkdir(exist_ok=True)
    with open(results_file, 'a+') as writer:
        for key in sorted(metrics):
            if key in ['log', 'progress_bar', 'preds']:
                continue
            val = metrics[key]
            if isinstance(val, torch.Tensor):
                val = val.item()
            msg = f'{key}: {val:.6f}\n'
            writer.write(msg)
    if not save_generations:
        return
    if 'preds' in metrics:
        content = '\n'.join(metrics['preds'])
        generations_file.open('w+').write(content)
