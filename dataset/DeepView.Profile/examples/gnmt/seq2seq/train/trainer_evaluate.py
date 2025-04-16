def evaluate(self, data_loader):
    """
        Sets model in eval mode, disables gradients, preallocates memory and
        runs validation on data provided by data_loader.

        :param data_loader: data loader
        """
    torch.set_grad_enabled(False)
    self.model.eval()
    torch.cuda.empty_cache()
    self.preallocate(data_loader, training=False)
    output = self.feed_data(data_loader, training=False)
    self.model.zero_grad()
    torch.cuda.empty_cache()
    return output
