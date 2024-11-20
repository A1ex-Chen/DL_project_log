def get_views(self, height, width, window_size=128, stride=64,
    random_jitter=False):
    height //= self.vae_scale_factor
    width //= self.vae_scale_factor
    num_blocks_height = int((height - window_size) / stride - 1e-06
        ) + 2 if height > window_size else 1
    num_blocks_width = int((width - window_size) / stride - 1e-06
        ) + 2 if width > window_size else 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int(i // num_blocks_width * stride)
        h_end = h_start + window_size
        w_start = int(i % num_blocks_width * stride)
        w_end = w_start + window_size
        if h_end > height:
            h_start = int(h_start + height - h_end)
            h_end = int(height)
        if w_end > width:
            w_start = int(w_start + width - w_end)
            w_end = int(width)
        if h_start < 0:
            h_end = int(h_end - h_start)
            h_start = 0
        if w_start < 0:
            w_end = int(w_end - w_start)
            w_start = 0
        if random_jitter:
            jitter_range = (window_size - stride) // 4
            w_jitter = 0
            h_jitter = 0
            if w_start != 0 and w_end != width:
                w_jitter = random.randint(-jitter_range, jitter_range)
            elif w_start == 0 and w_end != width:
                w_jitter = random.randint(-jitter_range, 0)
            elif w_start != 0 and w_end == width:
                w_jitter = random.randint(0, jitter_range)
            if h_start != 0 and h_end != height:
                h_jitter = random.randint(-jitter_range, jitter_range)
            elif h_start == 0 and h_end != height:
                h_jitter = random.randint(-jitter_range, 0)
            elif h_start != 0 and h_end == height:
                h_jitter = random.randint(0, jitter_range)
            h_start += h_jitter + jitter_range
            h_end += h_jitter + jitter_range
            w_start += w_jitter + jitter_range
            w_end += w_jitter + jitter_range
        views.append((h_start, h_end, w_start, w_end))
    return views
