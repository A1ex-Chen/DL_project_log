def update_labels(self, include_class: Optional[list]):
    """Update labels to include only these classes (optional)."""
    include_class_array = np.array(include_class).reshape(1, -1)
    for i in range(len(self.labels)):
        if include_class is not None:
            cls = self.labels[i]['cls']
            bboxes = self.labels[i]['bboxes']
            segments = self.labels[i]['segments']
            keypoints = self.labels[i]['keypoints']
            j = (cls == include_class_array).any(1)
            self.labels[i]['cls'] = cls[j]
            self.labels[i]['bboxes'] = bboxes[j]
            if segments:
                self.labels[i]['segments'] = [segments[si] for si, idx in
                    enumerate(j) if idx]
            if keypoints is not None:
                self.labels[i]['keypoints'] = keypoints[j]
        if self.single_cls:
            self.labels[i]['cls'][:, 0] = 0
