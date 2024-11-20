def parse_inference_input(to_predict):
    filenames = []
    image_formats = ['.jpg', '.jpeg', '.JPEG', '.JPG', '.png', '.PNG']
    if os.path.isdir(to_predict):
        filenames = [f for f in os.listdir(to_predict) if os.path.isfile(os
            .path.join(to_predict, f)) and os.path.splitext(f)[1] in
            image_formats]
    elif os.path.isfile(to_predict):
        filenames.append(to_predict)
    return filenames
