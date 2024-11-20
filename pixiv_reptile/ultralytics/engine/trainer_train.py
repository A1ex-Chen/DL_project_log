def train(self):
    """Allow device='', device=None on Multi-GPU systems to default to device=0."""
    if isinstance(self.args.device, str) and len(self.args.device):
        world_size = len(self.args.device.split(','))
    elif isinstance(self.args.device, (tuple, list)):
        world_size = len(self.args.device)
    elif torch.cuda.is_available():
        world_size = 1
    else:
        world_size = 0
    if world_size > 1 and 'LOCAL_RANK' not in os.environ:
        if self.args.rect:
            LOGGER.warning(
                "WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'"
                )
            self.args.rect = False
        if self.args.batch < 1.0:
            LOGGER.warning(
                "WARNING ⚠️ 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting default 'batch=16'"
                )
            self.args.batch = 16
        cmd, file = generate_ddp_command(world_size, self)
        try:
            LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except Exception as e:
            raise e
        finally:
            ddp_cleanup(self, str(file))
    else:
        self._do_train(world_size)
