def get_input_embeddings(self) ->nn.Module:
    return self.vision_model.embeddings.patch_embedding
