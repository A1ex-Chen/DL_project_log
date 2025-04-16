def __init__(self, model, criterion, opt_config, scheduler_config,
    print_freq=10, save_freq=1000, grad_clip=float('inf'), batch_first=
    False, save_info={}, save_path='.', train_iterations=0,
    checkpoint_filename='checkpoint%s.pth', keep_checkpoints=5, math='fp32',
    cuda=True, distributed=False, intra_epoch_eval=0, iter_size=1,
    translator=None, verbose=False):
    """
        Constructor for the Seq2SeqTrainer.

        :param model: model to train
        :param criterion: criterion (loss function)
        :param opt_config: dictionary with options for the optimizer
        :param scheduler_config: dictionary with options for the learning rate
            scheduler
        :param print_freq: prints short summary every 'print_freq' iterations
        :param save_freq: saves checkpoint every 'save_freq' iterations
        :param grad_clip: coefficient for gradient clipping
        :param batch_first: if True the model uses (batch,seq,feature) tensors,
            if false the model uses (seq, batch, feature)
        :param save_info: dict with additional state stored in each checkpoint
        :param save_path: path to the directiory for checkpoints
        :param train_iterations: total number of training iterations to execute
        :param checkpoint_filename: name of files with checkpoints
        :param keep_checkpoints: max number of checkpoints to keep
        :param math: arithmetic type
        :param cuda: if True use cuda, if False train on cpu
        :param distributed: if True run distributed training
        :param intra_epoch_eval: number of additional eval runs within each
            training epoch
        :param iter_size: number of iterations between weight updates
        :param translator: instance of Translator, runs inference on test set
        :param verbose: enables verbose logging
        """
    super(Seq2SeqTrainer, self).__init__()
    self.model = model
    self.criterion = criterion
    self.epoch = 0
    self.save_info = save_info
    self.save_path = save_path
    self.save_freq = save_freq
    self.save_counter = 0
    self.checkpoint_filename = checkpoint_filename
    self.checkpoint_counter = cycle(range(keep_checkpoints))
    self.opt_config = opt_config
    self.cuda = cuda
    self.distributed = distributed
    self.print_freq = print_freq
    self.batch_first = batch_first
    self.verbose = verbose
    self.loss = None
    self.translator = translator
    self.intra_epoch_eval = intra_epoch_eval
    self.iter_size = iter_size
    if cuda:
        self.model = self.model.cuda()
        self.criterion = self.criterion.cuda()
    if math == 'fp16':
        self.model = self.model.half()
    if distributed:
        self.model = DDP(self.model)
    if math == 'fp16':
        self.fp_optimizer = Fp16Optimizer(self.model, grad_clip)
        params = self.fp_optimizer.fp32_params
    elif math == 'fp32':
        self.fp_optimizer = Fp32Optimizer(self.model, grad_clip)
        params = self.model.parameters()
    opt_name = opt_config.pop('optimizer')
    self.optimizer = torch.optim.__dict__[opt_name](params, **opt_config)
    logging.info(f'Using optimizer: {self.optimizer}')
    self.scheduler = WarmupMultiStepLR(self.optimizer, train_iterations, **
        scheduler_config)
