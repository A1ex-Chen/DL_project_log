def train(self, start_iter: int, max_iter: int):
    """
        Args:
            start_iter, max_iter (int): See docs above
        """
    logger = logging.getLogger(__name__)
    logger.info('Starting training from iteration {}'.format(start_iter))
    self.iter = self.start_iter = start_iter
    self.max_iter = max_iter
    with EventStorage(start_iter) as self.storage:
        try:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.iter += 1
        except Exception:
            logger.exception('Exception during training:')
            raise
        finally:
            self.after_train()
