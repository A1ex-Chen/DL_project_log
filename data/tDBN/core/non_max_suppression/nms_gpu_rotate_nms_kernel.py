@cuda.jit('(int64, float32, float32[:], uint64[:])')
def rotate_nms_kernel(n_boxes, nms_overlap_thresh, dev_boxes, dev_mask):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 6,), dtype=numba.float32)
    dev_box_idx = threadsPerBlock * col_start + tx
    if tx < col_size:
        block_boxes[tx * 6 + 0] = dev_boxes[dev_box_idx * 6 + 0]
        block_boxes[tx * 6 + 1] = dev_boxes[dev_box_idx * 6 + 1]
        block_boxes[tx * 6 + 2] = dev_boxes[dev_box_idx * 6 + 2]
        block_boxes[tx * 6 + 3] = dev_boxes[dev_box_idx * 6 + 3]
        block_boxes[tx * 6 + 4] = dev_boxes[dev_box_idx * 6 + 4]
        block_boxes[tx * 6 + 5] = dev_boxes[dev_box_idx * 6 + 5]
    cuda.syncthreads()
    if tx < row_size:
        cur_box_idx = threadsPerBlock * row_start + tx
        t = 0
        start = 0
        if row_start == col_start:
            start = tx + 1
        for i in range(start, col_size):
            iou = devRotateIoU(dev_boxes[cur_box_idx * 6:cur_box_idx * 6 + 
                5], block_boxes[i * 6:i * 6 + 5])
            if iou > nms_overlap_thresh:
                t |= 1 << i
        col_blocks = n_boxes // threadsPerBlock + (n_boxes %
            threadsPerBlock > 0)
        dev_mask[cur_box_idx * col_blocks + col_start] = t
