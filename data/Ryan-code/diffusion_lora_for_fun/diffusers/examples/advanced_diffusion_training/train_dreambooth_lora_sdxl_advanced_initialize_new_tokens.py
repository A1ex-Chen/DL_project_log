def initialize_new_tokens(self, inserting_toks: List[str]):
    idx = 0
    for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
        assert isinstance(inserting_toks, list
            ), 'inserting_toks should be a list of strings.'
        assert all(isinstance(tok, str) for tok in inserting_toks
            ), 'All elements in inserting_toks should be strings.'
        self.inserting_toks = inserting_toks
        special_tokens_dict = {'additional_special_tokens': self.inserting_toks
            }
        tokenizer.add_special_tokens(special_tokens_dict)
        text_encoder.resize_token_embeddings(len(tokenizer))
        self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)
        std_token_embedding = (text_encoder.text_model.embeddings.
            token_embedding.weight.data.std())
        print(
            f"{idx} text encodedr's std_token_embedding: {std_token_embedding}"
            )
        text_encoder.text_model.embeddings.token_embedding.weight.data[self
            .train_ids] = torch.randn(len(self.train_ids), text_encoder.
            text_model.config.hidden_size).to(device=self.device).to(dtype=
            self.dtype) * std_token_embedding
        self.embeddings_settings[f'original_embeddings_{idx}'] = (text_encoder
            .text_model.embeddings.token_embedding.weight.data.clone())
        self.embeddings_settings[f'std_token_embedding_{idx}'
            ] = std_token_embedding
        inu = torch.ones((len(tokenizer),), dtype=torch.bool)
        inu[self.train_ids] = False
        self.embeddings_settings[f'index_no_updates_{idx}'] = inu
        print(self.embeddings_settings[f'index_no_updates_{idx}'].shape)
        idx += 1
