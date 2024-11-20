def _yield_batches(self, dataset: ExplorerDataset, data_info: dict, model:
    YOLO, exclude_keys: List[str]):
    """Generates batches of data for embedding, excluding specified keys."""
    for i in tqdm(range(len(dataset))):
        self.progress = float(i + 1) / len(dataset)
        batch = dataset[i]
        for k in exclude_keys:
            batch.pop(k, None)
        batch = sanitize_batch(batch, data_info)
        batch['vector'] = model.embed(batch['im_file'], verbose=False)[0
            ].detach().tolist()
        yield [batch]
