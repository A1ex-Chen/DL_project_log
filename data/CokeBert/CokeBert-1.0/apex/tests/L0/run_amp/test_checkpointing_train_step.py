def train_step(self, model, optimizer, data, loss_ids):
    optimizer.zero_grad()
    output = model(data)
    for idx in loss_ids:
        loss = output.mean()
        with amp.scale_loss(loss, optimizer, loss_id=idx) as scaled_loss:
            scaled_loss.backward(retain_graph=True)
    optimizer.step()
    return output
