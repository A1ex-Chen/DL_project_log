def training_step():
    for param in model.parameters():
        param.grad = None
    loss = model(self.x).sum()
    _amp_state.loss_scalers[0]._loss_scale = 4.0
    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    self.assertEqual(len([p.grad for p in model.parameters() if p.grad is not
        None]), 1)
    self.assertEqual(model.weight.grad.type(), model.weight.type())
    reference_grad = get_reference_grad(self.x, model.weight, model.ops)
    if model.weight.grad.type() == 'torch.cuda.HalfTensor':
        self.assertTrue(torch.allclose(model.weight.grad.float(),
            reference_grad))
    elif model.weight.grad.type() == 'torch.cuda.FloatTensor':
        self.assertTrue(torch.allclose(model.weight.grad.float(),
            reference_grad))
    else:
        raise RuntimeError('model.weight.grad.type = {}'.format(model.
            weight.grad.type()))
    model.weight.data -= 1.0
