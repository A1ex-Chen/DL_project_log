def forward(self, features_batch: Tensor) ->Tuple[Tensor, Tensor]:
    batch_size = features_batch.shape[0]
    features_batch = self._extractor(features_batch)
    anchor_objectnesses_batch = self.anchor_objectness(features_batch)
    anchor_transformers_batch = self.anchor_transformer(features_batch)
    anchor_objectnesses_batch = anchor_objectnesses_batch.permute(0, 2, 3, 1
        ).contiguous().view(batch_size, -1, 2)
    anchor_transformers_batch = anchor_transformers_batch.permute(0, 2, 3, 1
        ).contiguous().view(batch_size, -1, 4)
    return anchor_objectnesses_batch, anchor_transformers_batch
