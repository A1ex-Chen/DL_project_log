def load_state_dict(self, state_dict):
    if isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
        logger = logging.getLogger(__name__)
        logger.info('Loading scheduler from state_dict ...')
        self.scheduler.load_state_dict(state_dict)
