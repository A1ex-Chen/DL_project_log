def iteration(*inputs):
    optimizer.zero_grad()
    out = model(*inputs)
    out.backward()
    optimizer.step()
