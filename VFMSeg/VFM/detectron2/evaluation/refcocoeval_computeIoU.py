def computeIoU(self, imgId, catId):
    p = self.params
    if p.useCats:
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
    else:
        gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
        dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
    if len(gt) == 0 and len(dt) == 0:
        return []
    inds = np.argsort([(-d['score']) for d in dt], kind='mergesort')
    dt = [dt[i] for i in inds]
    if len(dt) > p.maxDets[-1]:
        dt = dt[0:p.maxDets[-1]]
    if p.iouType == 'segm':
        g = [g['segmentation'] for g in gt]
        d = [d['segmentation'] for d in dt]
    elif p.iouType == 'bbox':
        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]
    else:
        raise Exception('unknown iouType for iou computation')
    iscrowd = [int(o['iscrowd']) for o in gt]
    ious = maskUtils.iou(d, g, iscrowd)
    if p.iouType == 'bbox':
        g, d = g[0], d[0]
        g_bbox = [g[0], g[1], g[2] + g[0], g[3] + g[1]]
        d_bbox = [d[0], d[1], d[2] + d[0], d[3] + d[1]]
        g_bbox = torch.tensor(g_bbox).unsqueeze(0)
        d_bbox = torch.tensor(d_bbox).unsqueeze(0)
        iou, intersection, union = compute_bbox_iou(d_bbox, g_bbox)
    elif p.iouType == 'segm':
        g_segm = decode(g[0])
        d_segm = decode(d[0])
        g_segm = torch.tensor(g_segm).unsqueeze(0)
        d_segm = torch.tensor(d_segm).unsqueeze(0)
        iou, intersection, union = compute_mask_iou(d_segm, g_segm)
    else:
        raise Exception('unknown iouType for iou computation')
    iou, intersection, union = iou.item(), intersection.item(), union.item()
    self.total_intersection_area += intersection
    self.total_union_area += union
    self.iou_list.append(iou)
    return ious
