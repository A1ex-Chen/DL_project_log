@DISTORTION_REGISTRY.register('snow')
def snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8), (0.2, 0.3, 2, 0.5, 12, 4, 0.7), (
        0.55, 0.3, 4, 0.9, 12, 8, 0.7), (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
        (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0
    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 
        255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())
    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform
        (-135, -45))
    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.
        uint8), cv2.IMREAD_UNCHANGED) / 255.0
    snow_layer = snow_layer[..., np.newaxis]
    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.
        COLOR_RGB2GRAY)[..., np.newaxis] * 1.5 + 0.5)
    snow_layer = cv2.resize(snow_layer, (x.shape[1], x.shape[0]))[..., np.
        newaxis]
    snow_layer_rot = cv2.resize(np.rot90(snow_layer, k=2), (x.shape[1], x.
        shape[0]))[..., np.newaxis]
    return np.clip(x + snow_layer + snow_layer_rot, 0, 1) * 255
