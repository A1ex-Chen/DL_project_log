def masks2segments(masks, strategy='largest'):
    segments = []
    for x in masks.int().cpu().numpy().astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':
                c = np.array(c[np.array([len(x) for x in c]).argmax()]
                    ).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))
        segments.append(c.astype('float32'))
    return segments
