@torch.no_grad()
def get_text_embeddings(self, *args, **kwargs):
    return super().get_text_embeddings(*args, **kwargs)
