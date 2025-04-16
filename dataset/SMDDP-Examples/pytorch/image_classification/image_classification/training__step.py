def _step(input, target):
    input_var = Variable(input)
    target_var = Variable(target)
    with torch.no_grad(), autocast(enabled=use_amp):
        loss, output = model_and_loss(input_var, target_var)
        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))
        if dist.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data)
            prec1 = utils.reduce_tensor(prec1)
            prec5 = utils.reduce_tensor(prec5)
        else:
            reduced_loss = loss.data
    torch.cuda.synchronize()
    return reduced_loss, prec1, prec5
