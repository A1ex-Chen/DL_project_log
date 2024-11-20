def deepview_iteration_provider(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    def iteration(inputs, targets):
        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_fn(out, targets)
        loss.backward()
        optimizer.step()
    return iteration
