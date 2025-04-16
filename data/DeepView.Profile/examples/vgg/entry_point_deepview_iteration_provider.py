def deepview_iteration_provider(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    def iteration(*inputs):
        optimizer.zero_grad()
        out = model(*inputs)
        out.backward()
        optimizer.step()
    return iteration
