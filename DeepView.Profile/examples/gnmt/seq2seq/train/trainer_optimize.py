def optimize(self, data_loader):
    """
        Sets model in training mode, preallocates memory and runs training on
        data provided by data_loader.

        :param data_loader: data loader
        """
    torch.set_grad_enabled(True)
    self.model.train()
    torch.cuda.empty_cache()
    self.preallocate(data_loader, training=True)
    output = self.feed_data(data_loader, training=True)
    self.model.zero_grad()
    torch.cuda.empty_cache()
    return output
