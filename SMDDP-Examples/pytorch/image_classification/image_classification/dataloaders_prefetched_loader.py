def prefetched_loader(loader, num_classes, one_hot):
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(
        1, 3, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(
        1, 3, 1, 1)
    stream = torch.cuda.Stream()
    first = True
    for next_input, next_target in loader:
        with torch.cuda.stream(stream):
            next_input = next_input.cuda(non_blocking=True)
            next_target = next_target.cuda(non_blocking=True)
            next_input = next_input.float()
            if one_hot:
                next_target = expand(num_classes, torch.float, next_target)
            next_input = next_input.sub_(mean).div_(std)
        if not first:
            yield input, target
        else:
            first = False
        torch.cuda.current_stream().wait_stream(stream)
        input = next_input
        target = next_target
    yield input, target
