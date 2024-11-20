def get_bounding_boxes(self) ->Boxes:
    """
        Returns:
            Boxes: tight bounding boxes around polygon masks.
        """
    boxes = torch.zeros(len(self.polygons), 4, dtype=torch.float32)
    for idx, polygons_per_instance in enumerate(self.polygons):
        minxy = torch.as_tensor([float('inf'), float('inf')], dtype=torch.
            float32)
        maxxy = torch.zeros(2, dtype=torch.float32)
        for polygon in polygons_per_instance:
            coords = torch.from_numpy(polygon).view(-1, 2).to(dtype=torch.
                float32)
            minxy = torch.min(minxy, torch.min(coords, dim=0).values)
            maxxy = torch.max(maxxy, torch.max(coords, dim=0).values)
        boxes[idx, :2] = minxy
        boxes[idx, 2:] = maxxy
    return Boxes(boxes)
