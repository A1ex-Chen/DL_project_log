def lp_gather_features(pred, target, world_size=1, use_horovod=False):
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        with torch.no_grad():
            all_preds = hvd.allgather(pred)
            all_targets = hvd.allgath(target)
    else:
        gathered_preds = [torch.zeros_like(pred) for _ in range(world_size)]
        gathered_targets = [torch.zeros_like(target) for _ in range(world_size)
            ]
        dist.all_gather(gathered_preds, pred)
        dist.all_gather(gathered_targets, target)
        all_preds = torch.cat(gathered_preds, dim=0)
        all_targets = torch.cat(gathered_targets, dim=0)
    return all_preds, all_targets
