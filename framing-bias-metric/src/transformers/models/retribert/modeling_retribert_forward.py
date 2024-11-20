def forward(self, input_ids_query, attention_mask_query, input_ids_doc,
    attention_mask_doc, checkpoint_batch_size=-1):
    """
        Args:
            input_ids_query (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary for the queries in a batch.

                Indices can be obtained using :class:`~transformers.RetriBertTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask_query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            input_ids_doc (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary for the documents in a batch.
            attention_mask_doc (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on documents padding token indices.
            checkpoint_batch_size (:obj:`int`, `optional`, defaults to `:obj:`-1`):
                If greater than 0, uses gradient checkpointing to only compute sequence representation on
                :obj:`checkpoint_batch_size` examples at a time on the GPU. All query representations are still
                compared to all document representations in the batch.

        Return:
            :obj:`torch.FloatTensor`: The bidirectional cross-entropy loss obtained while trying to match each query to
            its corresponding document and each document to its corresponding query in the batch
        """
    device = input_ids_query.device
    q_reps = self.embed_questions(input_ids_query, attention_mask_query,
        checkpoint_batch_size)
    a_reps = self.embed_answers(input_ids_doc, attention_mask_doc,
        checkpoint_batch_size)
    compare_scores = torch.mm(q_reps, a_reps.t())
    loss_qa = self.ce_loss(compare_scores, torch.arange(compare_scores.
        shape[1]).to(device))
    loss_aq = self.ce_loss(compare_scores.t(), torch.arange(compare_scores.
        shape[0]).to(device))
    loss = (loss_qa + loss_aq) / 2
    return loss
