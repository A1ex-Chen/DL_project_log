@torch.no_grad()
def get_text_token_embeddings(self, *args, **kwargs):
    return super().get_text_token_embeddings(*args, **kwargs)
