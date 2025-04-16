def __init__(self, args=DEFAULT_CFG, _callbacks=None):
    """
        Initialize the Tuner with configurations.

        Args:
            args (dict, optional): Configuration for hyperparameter evolution.
        """
    self.space = args.pop('space', None) or {'lr0': (1e-05, 0.1), 'lrf': (
        0.0001, 0.1), 'momentum': (0.7, 0.98, 0.3), 'weight_decay': (0.0, 
        0.001), 'warmup_epochs': (0.0, 5.0), 'warmup_momentum': (0.0, 0.95),
        'box': (1.0, 20.0), 'cls': (0.2, 4.0), 'dfl': (0.4, 6.0), 'hsv_h':
        (0.0, 0.1), 'hsv_s': (0.0, 0.9), 'hsv_v': (0.0, 0.9), 'degrees': (
        0.0, 45.0), 'translate': (0.0, 0.9), 'scale': (0.0, 0.95), 'shear':
        (0.0, 10.0), 'perspective': (0.0, 0.001), 'flipud': (0.0, 1.0),
        'fliplr': (0.0, 1.0), 'bgr': (0.0, 1.0), 'mosaic': (0.0, 1.0),
        'mixup': (0.0, 1.0), 'copy_paste': (0.0, 1.0)}
    self.args = get_cfg(overrides=args)
    self.tune_dir = get_save_dir(self.args, name='tune')
    self.tune_csv = self.tune_dir / 'tune_results.csv'
    self.callbacks = _callbacks or callbacks.get_default_callbacks()
    self.prefix = colorstr('Tuner: ')
    callbacks.add_integration_callbacks(self)
    LOGGER.info(
        f"""{self.prefix}Initialized Tuner instance with 'tune_dir={self.tune_dir}'
{self.prefix}💡 Learn about tuning at https://docs.ultralytics.com/guides/hyperparameter-tuning"""
        )
