@cuda.jit('(int64, float32, float32[:, :], uint64[:])')
def nms_kernel_v2(n_boxes, nms_overlap_thresh, dev_boxes, dev_mask):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(threadsPerBlock, 5), dtype=numba
        .float32)
    dev_box_idx = threadsPerBlock * col_start + tx
    if tx < col_size:
        block_boxes[tx, 0] = dev_boxes[dev_box_idx, 0]
        block_boxes[tx, 1] = dev_boxes[dev_box_idx, 1]
        block_boxes[tx, 2] = dev_boxes[dev_box_idx, 2]
        block_boxes[tx, 3] = dev_boxes[dev_box_idx, 3]
        block_boxes[tx, 4] = dev_boxes[dev_box_idx, 4]
    cuda.syncthreads()
    if cuda.threadIdx.x < row_size:
        cur_box_idx = threadsPerBlock * row_start + cuda.threadIdx.x
        i = 0
        t = 0
        start = 0
        if row_start == col_start:
            start = tx + 1
        for i in range(start, col_size):
            if iou_device(dev_boxes[cur_box_idx], block_boxes[i]
                ) > nms_overlap_thresh:
                t |= 1 << i
        col_blocks = n_boxes // threadsPerBlock + (n_boxes %
            threadsPerBlock > 0)
        dev_mask[cur_box_idx * col_blocks + col_start] = t
