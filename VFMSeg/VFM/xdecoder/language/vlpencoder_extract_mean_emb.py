def extract_mean_emb(txts):
    tokens = self.tokenizer(txts, padding='max_length', truncation=True,
        max_length=self.max_token_num, return_tensors='pt')
    clss_embedding = self.forward_language((tokens['input_ids'].cuda(),
        tokens['attention_mask'].cuda()), norm=norm)
    clss_embedding = clss_embedding.mean(dim=0)
    clss_embedding /= clss_embedding.norm()
    return clss_embedding
