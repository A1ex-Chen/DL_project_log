def embed_questions(self, input_ids, attention_mask=None,
    checkpoint_batch_size=-1):
    q_reps = self.embed_sentences_checkpointed(input_ids, attention_mask,
        self.bert_query, checkpoint_batch_size)
    return self.project_query(q_reps)
