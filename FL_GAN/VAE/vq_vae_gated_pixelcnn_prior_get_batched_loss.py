def get_batched_loss(args, data_loader, model, loss_func, loss_triples=True):
    """
    Gets loss in a batched fashion.
    Input is data loader, model to produce output and loss function
    Assuming loss output is VLB, reconstruct_loss, KL
    """
    losses = [[], [], []] if loss_triples else []
    loop = tqdm.tqdm(data_loader, total=len(data_loader), leave=False)
    model.eval()
    for images, labels in loop:
        if loop.last_print_n > 0:
            break
        images = images.to(args.device)
        out = model(images)
        loss = loss_func(images, out)
        if loss_triples:
            losses[0].append(loss[0].cpu().item())
            losses[1].append(loss[1].cpu().item())
            losses[2].append(loss[2].cpu().item())
        else:
            losses.append(loss.cpu().item())
    losses = np.array(losses)
    if not loss_triples:
        return np.mean(losses)
    return np.mean(losses[0]), np.mean(losses[1]), np.mean(losses[2])
