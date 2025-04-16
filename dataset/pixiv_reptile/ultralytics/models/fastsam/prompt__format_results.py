@staticmethod
def _format_results(result, filter=0):
    """Formats detection results into list of annotations each containing ID, segmentation, bounding box, score and
        area.
        """
    annotations = []
    n = len(result.masks.data) if result.masks is not None else 0
    for i in range(n):
        mask = result.masks.data[i] == 1.0
        if torch.sum(mask) >= filter:
            annotation = {'id': i, 'segmentation': mask.cpu().numpy(),
                'bbox': result.boxes.data[i], 'score': result.boxes.conf[i]}
            annotation['area'] = annotation['segmentation'].sum()
            annotations.append(annotation)
    return annotations
