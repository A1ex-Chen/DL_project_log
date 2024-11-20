def create_loss_samples(self):
    num_loss_samples = int(100 * len(self.user_ids) ** 0.5)
    logger.debug('[BEGIN]building {} loss samples'.format(num_loss_samples))
    self.loss_samples = [t for t in self.draw(num_loss_samples)]
    logger.debug('[END]building {} loss samples'.format(num_loss_samples))
