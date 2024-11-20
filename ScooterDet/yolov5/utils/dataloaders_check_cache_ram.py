def check_cache_ram(self, safety_margin=0.1, prefix=''):
    b, gb = 0, 1 << 30
    n = min(self.n, 30)
    for _ in range(n):
        im = cv2.imread(random.choice(self.im_files))
        ratio = self.img_size / max(im.shape[0], im.shape[1])
        b += im.nbytes * ratio ** 2
    mem_required = b * self.n / n
    mem = psutil.virtual_memory()
    cache = mem_required * (1 + safety_margin) < mem.available
    if not cache:
        LOGGER.info(
            f"{prefix}{mem_required / gb:.1f}GB RAM required, {mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, {'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
    return cache
