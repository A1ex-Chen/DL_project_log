def save_embeddings(self, file_path: str):
    assert self.train_ids is not None, 'Initialize new tokens before saving embeddings.'
    tensors = {}
    idx_to_text_encoder_name = {(0): 'clip_l', (1): 'clip_g'}
    for idx, text_encoder in enumerate(self.text_encoders):
        assert text_encoder.text_model.embeddings.token_embedding.weight.data.shape[
            0] == len(self.tokenizers[0]), 'Tokenizers should be the same.'
        new_token_embeddings = (text_encoder.text_model.embeddings.
            token_embedding.weight.data[self.train_ids])
        tensors[idx_to_text_encoder_name[idx]] = new_token_embeddings
    save_file(tensors, file_path)
