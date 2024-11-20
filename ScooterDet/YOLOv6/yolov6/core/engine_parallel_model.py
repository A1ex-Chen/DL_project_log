@staticmethod
def parallel_model(args, model, device):
    dp_mode = device.type != 'cpu' and args.rank == -1
    if dp_mode and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use DDP instead.\n')
        model = torch.nn.DataParallel(model)
    ddp_mode = device.type != 'cpu' and args.rank != -1
    if ddp_mode:
        model = DDP(model, device_ids=[args.local_rank], output_device=args
            .local_rank)
    return model
