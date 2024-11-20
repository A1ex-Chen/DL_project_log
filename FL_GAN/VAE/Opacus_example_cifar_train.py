def train(args, model, train_loader, optimizer, privacy_engine, epoch, device):
    start_time = datetime.now()
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []
    if args.grad_sample_mode == 'no_op':
        from functorch import grad_and_value, make_functional, vmap
        fmodel, _fparams = make_functional(model)

        def compute_loss_stateless_model(params, sample, target):
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)
            predictions = fmodel(params, batch)
            loss = criterion(predictions, targets)
            return loss
        ft_compute_grad = grad_and_value(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
        params = list(model.parameters())
    for i, (images, target) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        target = target.to(device)
        output = model(images)
        if args.grad_sample_mode == 'no_op':
            per_sample_grads, per_sample_losses = ft_compute_sample_grad(params
                , images, target)
            per_sample_grads = [g.detach() for g in per_sample_grads]
            loss = torch.mean(per_sample_losses)
            for p, g in zip(params, per_sample_grads):
                p.grad_sample = g
        else:
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)
            top1_acc.append(acc1)
            loss.backward()
        losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        if i % args.print_freq == 0:
            if not args.disable_dp:
                epsilon = privacy_engine.accountant.get_epsilon(delta=args.
                    delta)
                print(
                    f'\tTrain Epoch: {epoch} \tLoss: {np.mean(losses):.6f} Acc@1: {np.mean(top1_acc):.6f} (ε = {epsilon:.2f}, δ = {args.delta})'
                    )
            else:
                print(
                    f'\tTrain Epoch: {epoch} \tLoss: {np.mean(losses):.6f} Acc@1: {np.mean(top1_acc):.6f} '
                    )
    train_duration = datetime.now() - start_time
    return train_duration
