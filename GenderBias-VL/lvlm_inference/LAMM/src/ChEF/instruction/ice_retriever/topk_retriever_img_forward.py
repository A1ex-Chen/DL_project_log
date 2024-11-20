def forward(self, dataloader, process_bar=False, information=''):
    print(information)
    res_list = []
    _dataloader = copy.deepcopy(dataloader)
    if process_bar:
        _dataloader = tqdm.tqdm(_dataloader)
    for _, entry in enumerate(_dataloader):
        with torch.no_grad():
            metadata = entry.pop('metadata')
            pixel_values = torch.stack(entry['pixel_values'], dim=0).to(self
                .device)
            res = self.model(pixel_values).last_hidden_state[:, 0].detach(
                ).cpu().numpy()
        res_list.extend([{'embed': r, 'metadata': m} for r, m in zip(res,
            metadata)])
    return res_list
