def embed_answers(self, input_ids, attention_mask=None,
    checkpoint_batch_size=-1):
    a_reps = self.embed_sentences_checkpointed(input_ids, attention_mask, 
        self.bert_query if self.bert_doc is None else self.bert_doc,
        checkpoint_batch_size)
    return self.project_doc(a_reps)
