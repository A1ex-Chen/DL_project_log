def video():
    while True:
        ret, frame = cap.read()
        net = cv2.dnn.readNet(modelWeights)
        detections = pre_process(frame, net)
        img = post_process(frame.copy(), detections)
        """
        Put efficiency information. The function getPerfProfile returns the overall time for inference(t)
        and the timings for each of the layers(in layersTimes).
        """
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.
            getTickFrequency())
        cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, (0, 0, 255
            ), THICKNESS, cv2.LINE_AA)
        cv2.imshow('Output', img)
        if cv2.waitKey(30) & 255 == ord('q'):
            break
