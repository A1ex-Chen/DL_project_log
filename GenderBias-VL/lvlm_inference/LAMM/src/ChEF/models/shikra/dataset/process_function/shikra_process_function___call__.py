def __call__(self, image: Image.Image, preprocessor: Dict[str, Any]) ->Dict[
    str, Any]:
    image_processor = preprocessor['image']
    if isinstance(image, (list, tuple)):
        image = image_processor.preprocess(image, return_tensors='pt')[
            'pixel_values']
        assert False, 'Shikra not support MultiImage'
    elif isinstance(image, PIL.Image.Image):
        image = image_processor.preprocess(image, return_tensors='pt')[
            'pixel_values'][0]
    else:
        if hasattr(image_processor, 'crop_size'):
            crop_size = image_processor.crop_size
            height, width = crop_size['height'], crop_size['width']
        else:
            raise ValueError("got empty image. and don't know how to pad")
        image = torch.zeros(3, height, width)
    return {'image': image}
