def train(epoch, loader, model, optimizer, device):
    loader = tqdm(loader)
    criterion = nn.CrossEntropyLoss()
    for i, (img, label) in enumerate(loader):
        model.zero_grad()
        img = img.to(device)
        out = model(img)
        loss = criterion(out, img)
        loss.backward()
        optimizer.step()
        _, pred = out.max(1)
        correct = (pred == img).float()
        accuracy = correct.sum() / img.numel()
        loader.set_description(
            f'epoch: {epoch + 1}; loss: {loss.item():.5f}; acc: {accuracy:.5f}'
            )
