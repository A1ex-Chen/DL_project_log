def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
    """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
    super().__init__(vis_processor, text_processor, vis_root, ann_paths)
