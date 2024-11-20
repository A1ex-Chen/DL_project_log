@DISTORTION_REGISTRY.register('jpeg_compression')
def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]
    w, h = x.shape[0], x.shape[1]
    x = PILImage.fromarray(x)
    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)
    return np.array(x)
