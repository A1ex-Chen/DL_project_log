def point_prompt(self, points, pointlabel):
    """Adjusts points on detected masks based on user input and returns the modified results."""
    if self.results[0].masks is not None:
        masks = self._format_results(self.results[0], 0)
        target_height, target_width = self.results[0].orig_shape
        h = masks[0]['segmentation'].shape[0]
        w = masks[0]['segmentation'].shape[1]
        if h != target_height or w != target_width:
            points = [[int(point[0] * w / target_width), int(point[1] * h /
                target_height)] for point in points]
        onemask = np.zeros((h, w))
        for annotation in masks:
            mask = annotation['segmentation'] if isinstance(annotation, dict
                ) else annotation
            for i, point in enumerate(points):
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 1:
                    onemask += mask
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 0:
                    onemask -= mask
        onemask = onemask >= 1
        self.results[0].masks.data = torch.tensor(np.array([onemask]))
    return self.results
