def permute_all_to_NHWA_K_not_concat(box_cls, box_delta, num_classes=80):
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes).reshape(-1,
        num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4).reshape(-1, 4) for x in
        box_delta]
    return box_cls_flattened, box_delta_flattened
