def inference(self, model, dataset):
    dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn
        =self.get_collate_fn(dataset))
    predictions = []
    for batch_idx, batch in tqdm(enumerate(dataloader), desc=
        'Running inference', total=len(dataloader)):
        predictions.extend(self.batch_inference(model, batch, dataset=
            dataset, batch_idx=batch_idx))
    self._after_inference_step(predictions)
