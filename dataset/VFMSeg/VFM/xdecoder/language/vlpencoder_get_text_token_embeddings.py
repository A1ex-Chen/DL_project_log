def get_text_token_embeddings(self, txts, name='default', token=False, norm
    =False):
    if not token:
        tokens = self.tokenizer(txts, padding='max_length', truncation=True,
            max_length=self.max_token_num, return_tensors='pt')
        tokens = {key: value.cuda() for key, value in tokens.items()}
    else:
        tokens = txts
    token_emb, class_emb = self.forward_language_token((tokens['input_ids'],
        tokens['attention_mask']), norm=norm)
    ret = {'tokens': tokens, 'token_emb': token_emb, 'class_emb': class_emb}
    setattr(self, '{}_token_embeddings'.format(name), ret)
    return ret
