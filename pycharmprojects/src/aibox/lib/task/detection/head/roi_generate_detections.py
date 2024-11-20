def generate_detections(self, proposal_bboxes: Tensor, proposal_classes:
    Tensor, proposal_transformers: Tensor, padded_image_width: int,
    padded_image_height: int) ->Tuple[Tensor, Tensor, Tensor]:
    transformer_normalize_std = self._transformer_normalize_std.to(device=
        proposal_bboxes.device)
    transformer_normalize_mean = self._transformer_normalize_mean.to(device
        =proposal_bboxes.device)
    proposal_transformers = (proposal_transformers *
        transformer_normalize_std + transformer_normalize_mean)
    proposal_bboxes = proposal_bboxes[:, None, :].repeat(1, self.
        _num_classes, 1)
    detection_bboxes = BBox.apply_transformer(proposal_bboxes,
        proposal_transformers)
    detection_bboxes = BBox.clip(detection_bboxes, left=0, top=0, right=
        padded_image_width, bottom=padded_image_height)
    detection_probs = F.softmax(input=proposal_classes, dim=-1)
    nms_bboxes = []
    nms_classes = []
    nms_probs = []
    for c in range(1, self._num_classes):
        class_bboxes = detection_bboxes[:, c, :]
        class_probs = detection_probs[:, c]
        kept_indices = nms(class_bboxes, class_probs, iou_threshold=self.
            _detection_nms_threshold)
        class_bboxes = class_bboxes[kept_indices]
        class_probs = class_probs[kept_indices]
        nms_bboxes.append(class_bboxes)
        nms_classes.append(torch.full((kept_indices.shape[0],), c, dtype=
            torch.int, device=kept_indices.device))
        nms_probs.append(class_probs)
    nms_bboxes = torch.cat(nms_bboxes, dim=0) if len(nms_bboxes
        ) > 0 else torch.empty(0, 4).to(detection_bboxes)
    nms_classes = torch.cat(nms_classes, dim=0) if len(nms_classes
        ) > 0 else torch.empty(0, 4, dtype=torch.int).to(nms_bboxes.device)
    nms_probs = torch.cat(nms_probs, dim=0) if len(nms_classes
        ) > 0 else torch.empty(0, 4).to(detection_probs)
    _, sorted_indices = torch.sort(nms_probs, dim=-1, descending=True)
    detection_bboxes = nms_bboxes[sorted_indices][:self.
        _num_detections_per_image]
    detection_classes = nms_classes[sorted_indices][:self.
        _num_detections_per_image]
    detection_probs = nms_probs[sorted_indices][:self._num_detections_per_image
        ]
    return detection_bboxes, detection_classes, detection_probs
