def gather_losses(losses_list):
    return [torch.mean(torch.stack(losses_list))]
