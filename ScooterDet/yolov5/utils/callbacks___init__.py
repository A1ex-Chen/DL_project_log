def __init__(self):
    self._callbacks = {'on_pretrain_routine_start': [],
        'on_pretrain_routine_end': [], 'on_train_start': [],
        'on_train_epoch_start': [], 'on_train_batch_start': [],
        'optimizer_step': [], 'on_before_zero_grad': [],
        'on_train_batch_end': [], 'on_train_epoch_end': [], 'on_val_start':
        [], 'on_val_batch_start': [], 'on_val_image_end': [],
        'on_val_batch_end': [], 'on_val_end': [], 'on_fit_epoch_end': [],
        'on_model_save': [], 'on_train_end': [], 'on_params_update': [],
        'teardown': []}
    self.stop_training = False
