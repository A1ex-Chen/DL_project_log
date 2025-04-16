def __init__(self, model, optimizer, args=None, model_ema=None, amp_scaler=
    None, checkpoint_prefix='checkpoint', recovery_prefix='recovery',
    checkpoint_dir='', recovery_dir='', decreasing=False, max_history=10,
    unwrap_fn=unwrap_model):
    self.model = model
    self.optimizer = optimizer
    self.args = args
    self.model_ema = model_ema
    self.amp_scaler = amp_scaler
    self.checkpoint_files = []
    self.best_epoch = None
    self.best_metric = None
    self.curr_recovery_file = ''
    self.last_recovery_file = ''
    self.checkpoint_dir = checkpoint_dir
    self.recovery_dir = recovery_dir
    self.save_prefix = checkpoint_prefix
    self.recovery_prefix = recovery_prefix
    self.extension = '.pth.tar'
    self.decreasing = decreasing
    self.cmp = operator.lt if decreasing else operator.gt
    self.max_history = max_history
    self.unwrap_fn = unwrap_fn
    assert self.max_history >= 1
