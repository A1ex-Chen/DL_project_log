def prepare_inputs(self, dataloader, device):
    """ load sample inputs to device """
    inputs = []
    for batch in dataloader:
        if type(batch) is torch.Tensor:
            batch_d = batch.to(device)
            batch_d = batch_d,
            inputs.append(batch_d)
        else:
            batch_d = []
            for x in batch:
                assert type(x) is torch.Tensor, 'input is not a tensor'
                batch_d.append(x.to(device))
            batch_d = tuple(batch_d)
            inputs.append(batch_d)
    return inputs
