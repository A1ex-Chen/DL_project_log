def forward(self, padded_image_batch: Tensor, gt_classes_batch: Tensor=None
    ) ->Union[Tensor, Tuple[Tensor, Tensor]]:
    raise NotImplementedError
