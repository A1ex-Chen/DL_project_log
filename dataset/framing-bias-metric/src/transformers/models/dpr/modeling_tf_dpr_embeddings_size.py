@property
def embeddings_size(self) ->int:
    if self.projection_dim > 0:
        return self.projection_dim
    return self.bert_model.config.hidden_size
