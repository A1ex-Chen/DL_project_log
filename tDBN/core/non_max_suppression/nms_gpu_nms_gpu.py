def nms_gpu(dets, nms_overlap_thresh, device_id=0):
    """nms in gpu.

    Args:
        dets ([type]): [description]
        nms_overlap_thresh ([type]): [description]
        device_id ([type], optional): Defaults to 0. [description]

    Returns:
        [type]: [description]
    """
    boxes_num = dets.shape[0]
    keep_out = np.zeros([boxes_num], dtype=np.int32)
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)
    boxes_host = dets[order, :]
    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    cuda.select_device(device_id)
    mask_host = np.zeros((boxes_num * col_blocks,), dtype=np.uint64)
    blockspergrid = div_up(boxes_num, threadsPerBlock), div_up(boxes_num,
        threadsPerBlock)
    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes_host.reshape([-1]), stream)
        mask_dev = cuda.to_device(mask_host, stream)
        nms_kernel[blockspergrid, threadsPerBlock, stream](boxes_num,
            nms_overlap_thresh, boxes_dev, mask_dev)
        mask_dev.copy_to_host(mask_host, stream=stream)
    num_out = nms_postprocess(keep_out, mask_host, boxes_num)
    keep = keep_out[:num_out]
    return list(order[keep])
