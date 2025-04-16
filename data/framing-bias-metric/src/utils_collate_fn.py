def collate_fn(self, batch) ->Dict[str, torch.Tensor]:
    """Call prepare_seq2seq_batch."""
    batch_encoding: Dict[str, torch.Tensor
        ] = self.tokenizer.prepare_seq2seq_batch([x['src_texts'] for x in
        batch], tgt_texts=[x['tgt_texts'] for x in batch], max_length=self.
        max_source_length, max_target_length=self.max_target_length,
        return_tensors='pt', **self.dataset_kwargs).data
    batch_encoding['ids'] = torch.tensor([x['id'] for x in batch])
    if self.mt:
        if self.extra_task in CLASSIFICATION_TASKS:
            extra_task_batch_encoding = self.tokenizer.prepare_seq2seq_batch([
                x['extra_task_text'] for x in batch], tgt_texts=['' for _ in
                batch], max_length=self.max_source_length,
                max_target_length=self.max_target_length, return_tensors=
                'pt', **self.dataset_kwargs).data
            extra_task_batch_input_ids = extra_task_batch_encoding['input_ids']
            extra_task_batch_attention_mask = extra_task_batch_encoding[
                'attention_mask']
            extra_task_batch_labels = torch.tensor([x['extra_task_label'] for
                x in batch], dtype=torch.long)
            batch_encoding['extra_task_text_input_ids'
                ] = extra_task_batch_input_ids
            batch_encoding['extra_task_batch_attention_mask'
                ] = extra_task_batch_attention_mask
            batch_encoding['extra_task_labels'] = extra_task_batch_labels
        elif self.extra_task in GENERATION_TASKS:
            extra_batch_encoding = self.tokenizer.prepare_seq2seq_batch([x[
                'extra_src_texts'] for x in batch], tgt_texts=[x[
                'extra_tgt_texts'] for x in batch], max_length=self.
                max_source_length, max_target_length=self.max_target_length,
                return_tensors='pt', **self.dataset_kwargs).data
            batch_encoding['extra_task_text_input_ids'] = extra_batch_encoding[
                'input_ids']
            batch_encoding['extra_task_batch_attention_mask'
                ] = extra_batch_encoding['attention_mask']
            batch_encoding['extra_task_labels'] = extra_batch_encoding['labels'
                ]
    return batch_encoding
