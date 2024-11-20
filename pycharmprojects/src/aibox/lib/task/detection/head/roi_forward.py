def forward(self, features: Tensor, pre_extract_transform: Callable=None,
    post_extract_transform: Callable=None) ->Tuple[Tensor, Tensor]:
    if pre_extract_transform:
        features = pre_extract_transform(features)
    features = self._extractor(features)
    if post_extract_transform:
        features = post_extract_transform(features)
    proposal_classes = self.proposal_class(features)
    proposal_transformers = self.proposal_transformer(features)
    proposal_classes = proposal_classes.view(features.shape[0], self.
        _num_classes)
    proposal_transformers = proposal_transformers.view(features.shape[0],
        self._num_classes, 4)
    return proposal_classes, proposal_transformers
