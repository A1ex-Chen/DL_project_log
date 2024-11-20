@smart_inference_mode()
def __call__(self, trainer=None, model=None):
    """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        """
    self.training = trainer is not None
    augment = self.args.augment and not self.training
    if self.training:
        self.device = trainer.device
        self.data = trainer.data
        self.args.half = self.device.type != 'cpu'
        model = trainer.ema.ema or trainer.model
        model = model.half() if self.args.half else model.float()
        self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
        self.args.plots &= (trainer.stopper.possible_stop or trainer.epoch ==
            trainer.epochs - 1)
        model.eval()
    else:
        callbacks.add_integration_callbacks(self)
        model = AutoBackend(weights=model or self.args.model, device=
            select_device(self.args.device, self.args.batch), dnn=self.args
            .dnn, data=self.args.data, fp16=self.args.half)
        self.device = model.device
        self.args.half = model.fp16
        stride, pt, jit, engine = (model.stride, model.pt, model.jit, model
            .engine)
        imgsz = check_imgsz(self.args.imgsz, stride=stride)
        if engine:
            self.args.batch = model.batch_size
        elif not pt and not jit:
            self.args.batch = 1
            LOGGER.info(
                f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models'
                )
        if str(self.args.data).split('.')[-1] in {'yaml', 'yml'}:
            self.data = check_det_dataset(self.args.data)
        elif self.args.task == 'classify':
            self.data = check_cls_dataset(self.args.data, split=self.args.split
                )
        else:
            raise FileNotFoundError(emojis(
                f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"
                ))
        if self.device.type in {'cpu', 'mps'}:
            self.args.workers = 0
        if not pt:
            self.args.rect = False
        self.stride = model.stride
        self.dataloader = self.dataloader or self.get_dataloader(self.data.
            get(self.args.split), self.args.batch)
        model.eval()
        model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))
    self.run_callbacks('on_val_start')
    dt = Profile(device=self.device), Profile(device=self.device), Profile(
        device=self.device), Profile(device=self.device)
    bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.
        dataloader))
    self.init_metrics(de_parallel(model))
    self.jdict = []
    for batch_i, batch in enumerate(bar):
        self.run_callbacks('on_val_batch_start')
        self.batch_i = batch_i
        with dt[0]:
            batch = self.preprocess(batch)
        with dt[1]:
            preds = model(batch['img'], augment=augment)
        with dt[2]:
            if self.training:
                self.loss += model.loss(batch, preds)[1]
        with dt[3]:
            preds = self.postprocess(preds)
        self.update_metrics(preds, batch)
        if self.args.plots and batch_i < 3:
            self.plot_val_samples(batch, batch_i)
            self.plot_predictions(batch, preds, batch_i)
        self.run_callbacks('on_val_batch_end')
    stats = self.get_stats()
    self.check_stats(stats)
    self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.
        dataset) * 1000.0 for x in dt)))
    self.finalize_metrics()
    self.print_results()
    self.run_callbacks('on_val_end')
    if self.training:
        model.float()
        results = {**stats, **trainer.label_loss_items(self.loss.cpu() /
            len(self.dataloader), prefix='val')}
        return {k: round(float(v), 5) for k, v in results.items()}
    else:
        LOGGER.info(
            'Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image'
             % tuple(self.speed.values()))
        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                LOGGER.info(f'Saving {f.name}...')
                json.dump(self.jdict, f)
            stats = self.eval_json(stats)
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats
