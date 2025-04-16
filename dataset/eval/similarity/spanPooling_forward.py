def forward(self, features: Dict[str, Tensor]):
    token_embeddings = features['token_embeddings']
    cls_token = features['cls_token_embeddings'
        ] if 'cls_token_embeddings' in features else token_embeddings[:, 0, :]
    attention_mask = features['attention_mask']
    output_vectors = []
    if self.pooling_mode_cls_token:
        output_vectors.append(cls_token)
    if self.pooling_mode_max_tokens:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1000000000.0
        max_over_time = torch.max(token_embeddings, 1)[0]
        output_vectors.append(max_over_time)
    if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        if 'token_weights_sum' in features:
            sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(
                sum_embeddings.size())
        else:
            sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-09)
        if self.pooling_mode_mean_tokens:
            output_vectors.append(sum_embeddings / sum_mask)
        if self.pooling_mode_mean_sqrt_len_tokens:
            output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
    if self.pooling_mode_span:
        last_emb = token_embeddings[:, -1, :]
        concat = torch.cat((cls_token, last_emb), 1)
        output_vectors.append(concat)
    output_vector = torch.cat(output_vectors, 1)
    features.update({'sentence_embedding': output_vector})
    return features
