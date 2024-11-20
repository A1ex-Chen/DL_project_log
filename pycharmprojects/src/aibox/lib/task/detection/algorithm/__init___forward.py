def forward(self, padded_image_batch: Tensor, gt_bboxes_batch: List[Tensor]
    =None, gt_classes_batch: List[Tensor]=None) ->Union[Tuple[Tensor,
    Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor],
    List[Tensor], List[Tensor], List[Tensor]]]:
    raise NotImplementedError
