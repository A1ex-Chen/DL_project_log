def hessian_vector_product(self, vector, data, target, r=0.01):
    """
        slightly touch vector value to estimate the gradient with respect to alpha
        refer to Eq. 7 for more details.
        :param vector: gradient.data of parameters theta
        :param x:
        :param target:
        :param r:
        :return:
        """
    R = r / F.flatten(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
        p.data.add_(R, v)
    loss = self.model.loss(data, target)
    grads_p = autograd.grad(loss, self.model.arch_parameters())
    for p, v in zip(self.model.parameters(), vector):
        p.data.sub_(2 * R, v)
    loss = self.model.loss(data, target)
    grads_n = autograd.grad(loss, self.model.arch_parameters())
    for p, v in zip(self.model.parameters(), vector):
        p.data.add_(R, v)
    h = [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
    return h
