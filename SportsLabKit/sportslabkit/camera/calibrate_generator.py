def generator():
    stab = Stabilizer()
    from sportslabkit.camera import Camera
    camera = Camera(media_path)
    for frame in camera:
        stab_frame = stab.stabilize(frame)
        if stab_frame is not None and stabilize:
            frame = stab_frame
        frame = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)
        yield frame
