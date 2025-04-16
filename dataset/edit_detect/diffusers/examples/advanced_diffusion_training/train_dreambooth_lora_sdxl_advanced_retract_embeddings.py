@torch.no_grad()
def retract_embeddings(self):
    for idx, text_encoder in enumerate(self.text_encoders):
        index_no_updates = self.embeddings_settings[f'index_no_updates_{idx}']
        text_encoder.text_model.embeddings.token_embedding.weight.data[
            index_no_updates] = self.embeddings_settings[
            f'original_embeddings_{idx}'][index_no_updates].to(device=
            text_encoder.device).to(dtype=text_encoder.dtype)
        std_token_embedding = self.embeddings_settings[
            f'std_token_embedding_{idx}']
        index_updates = ~index_no_updates
        new_embeddings = (text_encoder.text_model.embeddings.
            token_embedding.weight.data[index_updates])
        off_ratio = std_token_embedding / new_embeddings.std()
        new_embeddings = new_embeddings * off_ratio ** 0.1
        text_encoder.text_model.embeddings.token_embedding.weight.data[
            index_updates] = new_embeddings
