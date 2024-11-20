def backward_step_unrolled(self, x_train, target_train, x_valid,
    target_valid, eta, optimizer):
    """
        train on validate set based on update w_pi
        :param x_train:
        :param target_train:
        :param x_valid:
        :param target_valid:
        :param eta: 0.01, according to author's comments
        :param optimizer: theta optimizer
        :return:
        """
    unrolled_model = self.comp_unrolled_model(x_train, target_train, eta,
        optimizer)
    unrolled_loss = unrolled_model.loss(x_valid, target_valid)
    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self.hessian_vector_product(vector, x_train, target_train)
    for g, ig in zip(dalpha, implicit_grads):
        g.data.sub_(eta, ig.data)
    for v, g in zip(self.model.arch_parameters(), dalpha):
        if v.grad is None:
            v.grad = g.data
        else:
            v.grad.data.copy_(g.data)
