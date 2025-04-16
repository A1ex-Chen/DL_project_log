def apply_classifier(x, model, img, im0):
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):
        if d is not None and len(d):
            d = d.clone()
            b = xyxy2xywh(d[:, :4])
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
            b[:, 2:] = b[:, 2:] * 1.3 + 30
            d[:, :4] = xywh2xyxy(b).long()
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))
                im = im[:, :, ::-1].transpose(2, 0, 1)
                im = np.ascontiguousarray(im, dtype=np.float32)
                im /= 255.0
                ims.append(im)
            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)
            x[i] = x[i][pred_cls1 == pred_cls2]
    return x
