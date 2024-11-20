def predict(image_path, checkpoint, device, topk=5):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    rebuilt_model = load_checkpoint(checkpoint)
    rebuilt_model = rebuilt_model.to(device)
    processed_image = du.process_image(Image.open(image_path).convert('RGB'))
    processed_image = torch.from_numpy(np.array(processed_image))
    processed_image = processed_image.unsqueeze_(0)
    rebuilt_model.eval()
    processed_image = processed_image.to(device)
    with torch.no_grad():
        output = rebuilt_model.forward(processed_image)
    probabilities = torch.exp(output)
    probs = probabilities.topk(topk)[0]
    index = probabilities.topk(topk)[1]
    probs = np.array(probs)[0]
    index = np.array(index)[0]
    class_to_idx = rebuilt_model.class_to_idx
    inv_class_to_idx = {x: y for y, x in class_to_idx.items()}
    classes = []
    for element in index:
        classes += [inv_class_to_idx[element]]
    return probs, classes
