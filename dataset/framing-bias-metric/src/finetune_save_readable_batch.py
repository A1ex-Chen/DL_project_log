def save_readable_batch(self, batch: Dict[str, torch.Tensor]) ->Dict[str,
    List[str]]:
    """A debugging utility"""
    readable_batch = {k: (self.tokenizer.batch_decode(v.tolist()) if 'mask'
         not in k else v.shape) for k, v in batch.items()}
    save_json(readable_batch, Path(self.output_dir) / 'text_batch.json')
    save_json({k: v.tolist() for k, v in batch.items()}, Path(self.
        output_dir) / 'tok_batch.json')
    self.already_saved_batch = True
    return readable_batch
