def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    processing = transforms.Compose([transforms.Resize(256), transforms.
        CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[
        0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = processing(image)
    return image
