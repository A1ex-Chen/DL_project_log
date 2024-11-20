def set_epoch(self, epoch):
    """
        Sets current epoch index.
        Epoch index is used to seed RNG in __iter__() function.
        :param epoch: index of current epoch
        """
    self.epoch = epoch
