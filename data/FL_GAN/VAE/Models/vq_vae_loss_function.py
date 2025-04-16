def loss_function(self, *args, **kwargs) ->dict:
    """
        :param args:
        :param kwargs:
        :return:
        """
    recons = args[0]
    input = args[1]
    vq_loss = args[2]
    recons_loss = F.mse_loss(recons, input)
    loss = recons_loss + vq_loss
    return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'VQ_Loss':
        vq_loss}
