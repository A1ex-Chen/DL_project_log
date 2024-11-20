def train_iteration(model, i, mems, data_chunks, target_chunks, scaler,
    optimizer, device, delay_unscale, args):
    cpu = torch.device('cpu')
    data_i = data_chunks[i].contiguous()
    target_i = target_chunks[i].contiguous()
    if args.swap_mem and mems[i] is not None:
        mems[i] = mems[i].to(device, non_blocking=True)
    enable_autocast = args.fp16 and args.amp == 'pytorch'
    with torch.cuda.amp.autocast(enable_autocast):
        loss, mems[i] = model(data_i, target_i, mems[i])
        loss = loss.float().mean().type_as(loss) / args.batch_chunk
    if args.swap_mem and mems[i] is not None:
        mems[i] = mems[i].to(cpu, non_blocking=True)
    if args.fp16:
        if args.amp == 'pytorch':
            scaler.scale(loss).backward()
        elif args.amp == 'apex':
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale
                ) as scaled_loss:
                scaled_loss.backward()
    else:
        loss.backward()
    train_loss = loss.float().item()
    return train_loss
