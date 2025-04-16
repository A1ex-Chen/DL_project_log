def compute_metrics(self):
    """Use accumulated predictions and groundtruths to compute Average Precision."""
    overall_ret = OrderedDict()
    for ap_iou_thresh in self.ap_iou_thresh:
        ret_dict = OrderedDict()
        rec, prec, ap = eval_det_multiprocessing(self.pred_map_cls, self.
            gt_map_cls, ovthresh=ap_iou_thresh)
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(
                key)
            ret_dict['%s Average Precision' % clsname] = ap[key]
        ap_vals = np.array(list(ap.values()), dtype=np.float32)
        ap_vals[np.isnan(ap_vals)] = 0
        ret_dict['mAP'] = ap_vals.mean()
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(
                key)
            try:
                ret_dict['%s Recall' % clsname] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall' % clsname] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        overall_ret[ap_iou_thresh] = ret_dict
    return overall_ret
