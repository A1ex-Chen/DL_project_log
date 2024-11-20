def add_tokens(self, tokens):
    if len(tokens) == 0:
        return
    self.llama_tokenizer.add_tokens(['<XSFQ/>'], special_tokens=True)
    num_new_tokens = self.llama_tokenizer.add_tokens(tokens, special_tokens
        =True)
    self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
    if num_new_tokens > 0:
        input_embeddings = self.llama_model.get_input_embeddings().weight.data
        output_embeddings = self.llama_model.get_output_embeddings(
            ).weight.data
        input_embedding_avg = input_embeddings[:-num_new_tokens].mean(dim=0,
            keepdim=True)
        output_embedding_avg = output_embeddings[:-num_new_tokens].mean(dim
            =0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embedding_avg
        output_embeddings[-num_new_tokens:] = output_embedding_avg
