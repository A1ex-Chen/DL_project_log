@abstractmethod
def _train_epoch(self, epoch):
    """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
    raise NotImplementedError
