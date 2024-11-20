def _toMask(anns, coco):
    for ann in anns:
        rle = coco.annToRLE(ann)
        ann['segmentation'] = rle
