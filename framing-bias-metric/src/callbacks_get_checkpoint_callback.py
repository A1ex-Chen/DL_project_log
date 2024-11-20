def get_checkpoint_callback(output_dir, metric, save_top_k=1,
    lower_is_better=False):
    """Saves the best model by validation ROUGE2 score."""
    if metric == 'rouge2':
        exp = '{val_avg_rouge2:.4f}-{step_count}'
    elif metric == 'bleu':
        exp = '{val_avg_bleu:.4f}-{step_count}'
    elif metric == 'loss':
        exp = '{val_avg_loss:.4f}-{step_count}'
    else:
        raise NotImplementedError(
            f'seq2seq callbacks only support rouge2, bleu and loss, got {metric}, You can make your own by adding to this function.'
            )
    checkpoint_callback = ModelCheckpoint(dirpath=output_dir, filename=exp,
        monitor=f'val_{metric}', mode='min' if 'loss' in metric else 'max',
        save_top_k=save_top_k)
    return checkpoint_callback
