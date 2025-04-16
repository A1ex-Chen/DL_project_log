def draw_dataset_dict(self, dic):
    """
        Draw annotations/segmentaions in Detectron2 Dataset format.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
    annos = dic.get('annotations', None)
    if annos:
        if 'segmentation' in annos[0]:
            masks = [x['segmentation'] for x in annos]
        else:
            masks = None
        if 'keypoints' in annos[0]:
            keypts = [x['keypoints'] for x in annos]
            keypts = np.array(keypts).reshape(len(annos), -1, 3)
        else:
            keypts = None
        boxes = [(BoxMode.convert(x['bbox'], x['bbox_mode'], BoxMode.
            XYXY_ABS) if len(x['bbox']) == 4 else x['bbox']) for x in annos]
        colors = None
        category_ids = [x['category_id'] for x in annos]
        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get(
            'thing_colors'):
            colors = [self._jitter([(x / 255) for x in self.metadata.
                thing_colors[c]]) for c in category_ids]
        names = self.metadata.get('thing_classes', None)
        labels = _create_text_labels(category_ids, scores=None, class_names
            =names, is_crowd=[x.get('iscrowd', 0) for x in annos])
        self.overlay_instances(labels=labels, boxes=boxes, masks=masks,
            keypoints=keypts, assigned_colors=colors)
    sem_seg = dic.get('sem_seg', None)
    if sem_seg is None and 'sem_seg_file_name' in dic:
        with PathManager.open(dic['sem_seg_file_name'], 'rb') as f:
            sem_seg = Image.open(f)
            sem_seg = np.asarray(sem_seg, dtype='uint8')
    if sem_seg is not None:
        self.draw_sem_seg(sem_seg, area_threshold=0, alpha=0.5)
    pan_seg = dic.get('pan_seg', None)
    if pan_seg is None and 'pan_seg_file_name' in dic:
        with PathManager.open(dic['pan_seg_file_name'], 'rb') as f:
            pan_seg = Image.open(f)
            pan_seg = np.asarray(pan_seg)
            from panopticapi.utils import rgb2id
            pan_seg = rgb2id(pan_seg)
    if pan_seg is not None:
        segments_info = dic['segments_info']
        pan_seg = torch.tensor(pan_seg)
        self.draw_panoptic_seg(pan_seg, segments_info, area_threshold=0,
            alpha=0.5)
    return self.output
