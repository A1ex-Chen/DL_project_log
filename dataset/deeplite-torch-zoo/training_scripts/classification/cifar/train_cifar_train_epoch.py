def train_epoch(model, train_loader, optimizer, epoch, writer, dryrun=False):
    model.train()
    total_loss = []
    for data, target in tqdm(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        prediction = model(data)
        loss = F.nll_loss(F.log_softmax(prediction, dim=1), target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        if dryrun:
            break
    avg_loss = sum(total_loss) / len(total_loss)
    writer.add_scalar('train/avg_loss', avg_loss, epoch)
    LOGGER.info(f'Epoch: {epoch}:')
    LOGGER.info(f'Train Set: Average Loss: {avg_loss:.2f}')
