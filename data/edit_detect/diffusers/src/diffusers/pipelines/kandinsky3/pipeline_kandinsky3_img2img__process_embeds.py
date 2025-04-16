def _process_embeds(self, embeddings, attention_mask, cut_context):
    if cut_context:
        embeddings[attention_mask == 0] = torch.zeros_like(embeddings[
            attention_mask == 0])
        max_seq_length = attention_mask.sum(-1).max() + 1
        embeddings = embeddings[:, :max_seq_length]
        attention_mask = attention_mask[:, :max_seq_length]
    return embeddings, attention_mask
