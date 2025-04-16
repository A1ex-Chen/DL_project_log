@property
def embeddings_size(self) ->int:
    if self.projection_dim > 0:
        return self.encode_proj.out_features
    return self.bert_model.config.hidden_size
