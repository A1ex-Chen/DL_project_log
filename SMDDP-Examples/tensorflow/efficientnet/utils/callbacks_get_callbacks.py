def get_callbacks(model_checkpoint: bool=True, include_tensorboard: bool=
    True, time_history: bool=True, track_lr: bool=True, write_model_weights:
    bool=True, initial_step: int=0, batch_size: int=0, log_steps: int=100,
    model_dir: str=None, save_checkpoint_freq: int=0, logger=None) ->List[tf
    .keras.callbacks.Callback]:
    """Get all callbacks."""
    model_dir = model_dir or ''
    callbacks = []
    if model_checkpoint and sdp.rank() == 0:
        ckpt_full_path = os.path.join(model_dir, 'model.ckpt-{epoch:04d}')
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,
            save_weights_only=True, verbose=1, save_freq=save_checkpoint_freq))
    if time_history and logger is not None and sdp.rank() == 0:
        callbacks.append(TimeHistory(batch_size, log_steps, logdir=
            model_dir if include_tensorboard else None, logger=logger))
    if include_tensorboard:
        callbacks.append(CustomTensorBoard(log_dir=model_dir, track_lr=
            track_lr, initial_step=initial_step, write_images=
            write_model_weights))
    return callbacks
