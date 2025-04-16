def get_train_step(model_and_loss, optimizer, scaler, use_amp=False,
    batch_size_multiplier=1):

    def _step(input, target, optimizer_step=True):
        input_var = Variable(input)
        target_var = Variable(target)
        with autocast(enabled=use_amp):
            loss, output = model_and_loss(input_var, target_var)
            loss /= batch_size_multiplier
            if dist.is_initialized():
                reduced_loss = utils.reduce_tensor(loss.data)
            else:
                reduced_loss = loss.data
        scaler.scale(loss).backward()
        if optimizer_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        torch.cuda.synchronize()
        return reduced_loss
    return _step
