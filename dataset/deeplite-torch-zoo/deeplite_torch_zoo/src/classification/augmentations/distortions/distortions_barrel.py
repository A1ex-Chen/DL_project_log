@DISTORTION_REGISTRY.register('barrel')
def barrel(x, severity=1):
    c = [(0, 0.03, 0.03), (0.05, 0.05, 0.05), (0.1, 0.1, 0.1), (0.2, 0.2, 
        0.2), (0.1, 0.3, 0.6)][severity - 1]
    output = BytesIO()
    PILImage.fromarray(x).save(output, format='PNG')
    x = WandImage(blob=output.getvalue())
    x.distort('barrel', c)
    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8), cv2.
        IMREAD_UNCHANGED)
    if x.shape != (224, 224):
        return np.clip(x[..., [2, 1, 0]], 0, 255)
    else:
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)
