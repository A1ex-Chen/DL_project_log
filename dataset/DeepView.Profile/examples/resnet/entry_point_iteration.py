def iteration(inputs, targets):
    optimizer.zero_grad()
    out = model(inputs)
    loss = loss_fn(out, targets)
    loss.backward()
    optimizer.step()
