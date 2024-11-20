def __init__(self, keywords, tokenizer, input_ids):
    self.keywords = keywords
    self.keyword_ids = []
    self.max_keyword_len = 0
    for keyword in keywords:
        cur_keyword_ids = tokenizer(keyword).input_ids
        if len(cur_keyword_ids) > 1 and cur_keyword_ids[0
            ] == tokenizer.bos_token_id:
            cur_keyword_ids = cur_keyword_ids[1:]
        if len(cur_keyword_ids) > self.max_keyword_len:
            self.max_keyword_len = len(cur_keyword_ids)
        self.keyword_ids.append(torch.tensor(cur_keyword_ids))
    self.tokenizer = tokenizer
    self.start_len = input_ids.shape[1]
