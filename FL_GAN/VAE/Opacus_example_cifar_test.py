def test(args, model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []
    with torch.no_grad():
        for images, target in tqdm(test_loader):
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)
            losses.append(loss.item())
            top1_acc.append(acc1)
    top1_avg = np.mean(top1_acc)
    print(f'\tTest set:Loss: {np.mean(losses):.6f} Acc@1: {top1_avg:.6f} ')
    return np.mean(top1_acc)
