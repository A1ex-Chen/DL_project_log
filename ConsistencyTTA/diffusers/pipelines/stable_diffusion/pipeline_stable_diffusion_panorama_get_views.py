def get_views(self, panorama_height, panorama_width, window_size=64, stride=8):
    panorama_height /= 8
    panorama_width /= 8
    num_blocks_height = (panorama_height - window_size) // stride + 1
    num_blocks_width = (panorama_width - window_size) // stride + 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int(i // num_blocks_width * stride)
        h_end = h_start + window_size
        w_start = int(i % num_blocks_width * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views
