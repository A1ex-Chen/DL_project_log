def __init__(self, model, grad_clip=None):
    """
        Constructor for the Fp32Optimizer

        :param model: model
        :param grad_clip: coefficient for gradient clipping, max L2 norm of the
            gradients
        """
    self.initialize_model(model)
    self.grad_clip = grad_clip
