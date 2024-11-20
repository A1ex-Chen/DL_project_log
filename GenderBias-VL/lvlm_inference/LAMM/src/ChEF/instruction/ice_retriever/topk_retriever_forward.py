def forward(self, dataloader, process_bar=False, information=''):
    print(information)
    res_list = []
    _dataloader = copy.deepcopy(dataloader)
    if process_bar:
        _dataloader = tqdm.tqdm(_dataloader)
    for _, entry in enumerate(_dataloader):
        with torch.no_grad():
            metadata = entry.pop('metadata')
            raw_text = self.tokenizer.batch_decode(entry['input_ids'],
                skip_special_tokens=True, verbose=False)
            res = self.model.encode(raw_text, show_progress_bar=False)
        res_list.extend([{'embed': r, 'metadata': m} for r, m in zip(res,
            metadata)])
    return res_list
