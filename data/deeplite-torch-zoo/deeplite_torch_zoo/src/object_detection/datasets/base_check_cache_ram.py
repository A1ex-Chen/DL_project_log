def check_cache_ram(self, safety_margin=0.5):
    """Check image caching requirements vs available memory."""
    b, gb = 0, 1 << 30
    n = min(self.ni, 30)
    for _ in range(n):
        im = cv2.imread(random.choice(self.im_files))
        ratio = self.imgsz / max(im.shape[0], im.shape[1])
        b += im.nbytes * ratio ** 2
    mem_required = b * self.ni / n * (1 + safety_margin)
    mem = psutil.virtual_memory()
    cache = mem_required < mem.available
    if not cache:
        LOGGER.info(
            f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images with {int(safety_margin * 100)}% safety margin but only {mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, {'caching images ✅' if cache else 'not caching images ⚠️'}"
            )
    return cache
