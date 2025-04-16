def _encode(self, batch) ->Dict[str, torch.Tensor]:
    batch_encoding = self.tokenizer.prepare_seq2seq_batch([x['src_texts'] for
        x in batch], tgt_texts=[x['tgt_texts'] for x in batch], max_length=
        self.data_args.max_source_length, max_target_length=self.data_args.
        max_target_length, padding='max_length' if self.tpu_num_cores is not
        None else 'longest', return_tensors='pt', **self.dataset_kwargs)
    return batch_encoding.data
