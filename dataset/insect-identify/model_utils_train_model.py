def train_model(model, trainloader, set_epochs, validloader, learning_rate,
    device, choosen_architecture):
    """ Train the given model
	"""
    criterion = nn.CrossEntropyLoss()
    logger = Logger('logs')
    """feature = list(model.features)[:30]
    for layer in feature[:25]:
        for param in layer.parameters():
            param.requires_grad = False
    """
    frozen = 0
    for child in model.children():
        frozen += 1
        if frozen < 8:
            for param in child.parameters():
                param.requires_grad = False
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.
        parameters()), lr=learning_rate)
    model.to(device)
    epochs = set_epochs
    steps = 0
    running_x = [[]]
    running_y = [[]]
    running_label = [[]]
    running_loss_it = 0
    print_every = 1
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss_it += loss.item()
            if steps % print_every == 0:
                accuracy = 0
                model.eval()
                validloss = 0
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        validloss += batch_loss
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)
                            ).item()
                        logps = list(logps.cpu().numpy())
                        labels = list(labels.cpu().numpy())
                        running_x.extend(logps)
                        running_label.extend(labels)
                print(
                    f'Epoch {epoch + 1}/{epochs}.. Train loss: {running_loss_it / print_every:.3f}.. Validation loss: {validloss / len(validloader):.3f}.. Validation accuracy: {accuracy / len(validloader):.3f}'
                    )
                evaluation_metrics = [(choosen_architecture + '_Train loss',
                    running_loss_it / print_every), (choosen_architecture +
                    '_Validation loss', validloss / len(validloader)), (
                    choosen_architecture + '_Validation accuracy', accuracy /
                    len(validloader))]
                logger.list_of_scalars_summary(evaluation_metrics, steps)
                running_loss_it = 0
                model.train()
    return model, epochs, optimizer
