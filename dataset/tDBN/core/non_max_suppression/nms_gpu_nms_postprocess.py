@numba.jit(nopython=True)
def nms_postprocess(keep_out, mask_host, boxes_num):
    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    remv = np.zeros(col_blocks, dtype=np.uint64)
    num_to_keep = 0
    for i in range(boxes_num):
        nblock = i // threadsPerBlock
        inblock = i % threadsPerBlock
        mask = np.array(1 << inblock, dtype=np.uint64)
        if not remv[nblock] & mask:
            keep_out[num_to_keep] = i
            num_to_keep += 1
            for j in range(nblock, col_blocks):
                remv[j] |= mask_host[i * col_blocks + j]
    return num_to_keep
