def pre_process(input_image, net):
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH,
        INPUT_HEIGHT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    return outputs
