def _cityscapes_files_to_dict(files, from_json, to_polygons):
    """
    Parse cityscapes annotation files to a instance segmentation dataset dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        A dict in Detectron2 Dataset format.
    """
    from cityscapesscripts.helpers.labels import id2label, name2label
    image_file, instance_id_file, _, json_file = files
    annos = []
    if from_json:
        from shapely.geometry import MultiPolygon, Polygon
        with PathManager.open(json_file, 'r') as f:
            jsonobj = json.load(f)
        ret = {'file_name': image_file, 'image_id': os.path.basename(
            image_file), 'height': jsonobj['imgHeight'], 'width': jsonobj[
            'imgWidth']}
        polygons_union = Polygon()
        for obj in jsonobj['objects'][::-1]:
            if 'deleted' in obj:
                continue
            label_name = obj['label']
            try:
                label = name2label[label_name]
            except KeyError:
                if label_name.endswith('group'):
                    label = name2label[label_name[:-len('group')]]
                else:
                    raise
            if label.id < 0:
                continue
            poly_coord = np.asarray(obj['polygon'], dtype='f4') + 0.5
            poly = Polygon(poly_coord).buffer(0.5, resolution=4)
            if not label.hasInstances or label.ignoreInEval:
                polygons_union = polygons_union.union(poly)
                continue
            poly_wo_overlaps = poly.difference(polygons_union)
            if poly_wo_overlaps.is_empty:
                continue
            polygons_union = polygons_union.union(poly)
            anno = {}
            anno['iscrowd'] = label_name.endswith('group')
            anno['category_id'] = label.id
            if isinstance(poly_wo_overlaps, Polygon):
                poly_list = [poly_wo_overlaps]
            elif isinstance(poly_wo_overlaps, MultiPolygon):
                poly_list = poly_wo_overlaps.geoms
            else:
                raise NotImplementedError('Unknown geometric structure {}'.
                    format(poly_wo_overlaps))
            poly_coord = []
            for poly_el in poly_list:
                poly_coord.append(list(chain(*poly_el.exterior.coords)))
            anno['segmentation'] = poly_coord
            xmin, ymin, xmax, ymax = poly_wo_overlaps.bounds
            anno['bbox'] = xmin, ymin, xmax, ymax
            anno['bbox_mode'] = BoxMode.XYXY_ABS
            annos.append(anno)
    else:
        with PathManager.open(instance_id_file, 'rb') as f:
            inst_image = np.asarray(Image.open(f), order='F')
        flattened_ids = np.unique(inst_image[inst_image >= 24])
        ret = {'file_name': image_file, 'image_id': os.path.basename(
            image_file), 'height': inst_image.shape[0], 'width': inst_image
            .shape[1]}
        for instance_id in flattened_ids:
            label_id = (instance_id // 1000 if instance_id >= 1000 else
                instance_id)
            label = id2label[label_id]
            if not label.hasInstances or label.ignoreInEval:
                continue
            anno = {}
            anno['iscrowd'] = instance_id < 1000
            anno['category_id'] = label.id
            mask = np.asarray(inst_image == instance_id, dtype=np.uint8,
                order='F')
            inds = np.nonzero(mask)
            ymin, ymax = inds[0].min(), inds[0].max()
            xmin, xmax = inds[1].min(), inds[1].max()
            anno['bbox'] = xmin, ymin, xmax, ymax
            if xmax <= xmin or ymax <= ymin:
                continue
            anno['bbox_mode'] = BoxMode.XYXY_ABS
            if to_polygons:
                contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE)[-2]
                polygons = [c.reshape(-1).tolist() for c in contours if len
                    (c) >= 3]
                if len(polygons) == 0:
                    continue
                anno['segmentation'] = polygons
            else:
                anno['segmentation'] = mask_util.encode(mask[:, :, None])[0]
            annos.append(anno)
    ret['annotations'] = annos
    return ret
