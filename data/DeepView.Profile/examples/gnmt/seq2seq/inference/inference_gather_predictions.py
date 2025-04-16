def gather_predictions(preds):
    world_size = get_world_size()
    if world_size > 1:
        all_preds = [preds.new(preds.size(0), preds.size(1)) for i in range
            (world_size)]
        dist.all_gather(all_preds, preds)
        preds = torch.cat(all_preds)
    return preds
