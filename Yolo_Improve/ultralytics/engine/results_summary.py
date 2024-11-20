def summary(self, normalize=False, decimals=5):
    """Convert inference results to a summarized dictionary with optional normalization for box coordinates."""
    results = []
    if self.probs is not None:
        class_id = self.probs.top1
        results.append({'name': self.names[class_id], 'class': class_id,
            'confidence': round(self.probs.top1conf.item(), decimals)})
        return results
    is_obb = self.obb is not None
    data = self.obb if is_obb else self.boxes
    h, w = self.orig_shape if normalize else (1, 1)
    for i, row in enumerate(data):
        class_id, conf = int(row.cls), round(row.conf.item(), decimals)
        box = (row.xyxyxyxy if is_obb else row.xyxy).squeeze().reshape(-1, 2
            ).tolist()
        xy = {}
        for j, b in enumerate(box):
            xy[f'x{j + 1}'] = round(b[0] / w, decimals)
            xy[f'y{j + 1}'] = round(b[1] / h, decimals)
        result = {'name': self.names[class_id], 'class': class_id,
            'confidence': conf, 'box': xy}
        if data.is_track:
            result['track_id'] = int(row.id.item())
        if self.masks:
            result['segments'] = {'x': (self.masks.xy[i][:, 0] / w).round(
                decimals).tolist(), 'y': (self.masks.xy[i][:, 1] / h).round
                (decimals).tolist()}
        if self.keypoints is not None:
            x, y, visible = self.keypoints[i].data[0].cpu().unbind(dim=1)
            result['keypoints'] = {'x': (x / w).numpy().round(decimals).
                tolist(), 'y': (y / h).numpy().round(decimals).tolist(),
                'visible': visible.numpy().round(decimals).tolist()}
        results.append(result)
    return results
