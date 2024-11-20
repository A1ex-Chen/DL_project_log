def test(model, test_loader, epoch, writer):
    model.eval()
    loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            prediction = model(data)
            loss += F.nll_loss(F.log_softmax(prediction, dim=1), target,
                reduction='sum')
            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    loss /= len(test_loader.dataset)
    percentage_correct = 100.0 * correct / len(test_loader.dataset)
    LOGGER.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
        .format(loss, correct, len(test_loader.dataset), percentage_correct))
    writer.add_scalar('test/avg_loss', loss, epoch)
    writer.add_scalar('test/accuracy', percentage_correct, epoch)
    return loss, percentage_correct
