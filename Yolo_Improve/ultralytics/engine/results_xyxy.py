@property
@lru_cache(maxsize=2)
def xyxy(self):
    """
        Convert the oriented bounding boxes (OBB) to axis-aligned bounding boxes in xyxy format (x1, y1, x2, y2).

        Returns:
            (torch.Tensor | numpy.ndarray): Axis-aligned bounding boxes in xyxy format with shape (num_boxes, 4).

        Example:
            ```python
            import torch
            from ultralytics import YOLO

            model = YOLO('yolov8n.pt')
            results = model('path/to/image.jpg')
            for result in results:
                obb = result.obb
                if obb is not None:
                    xyxy_boxes = obb.xyxy
                    # Do something with xyxy_boxes
            ```

        Note:
            This method is useful to perform operations that require axis-aligned bounding boxes, such as IoU
            calculation with non-rotated boxes. The conversion approximates the OBB by the minimal enclosing rectangle.
        """
    x = self.xyxyxyxy[..., 0]
    y = self.xyxyxyxy[..., 1]
    return torch.stack([x.amin(1), y.amin(1), x.amax(1), y.amax(1)], -1
        ) if isinstance(x, torch.Tensor) else np.stack([x.min(1), y.min(1),
        x.max(1), y.max(1)], -1)
