def call_SEEM(seem_model, image_path=None, np_image=None, pil_image=None,
    mapping=None, prompt=None):
    if prompt is not None:
        task = ['stroke']
    else:
        task = []
    if np_image:
        image = Image.fromarray(np_image)
        return inference(seem_model, image, task, (mapping, prompt))
    if image_path:
        image = Image.open(image_path)
        return inference(seem_model, image, task, (mapping, prompt))
    if pil_image:
        return inference(seem_model, pil_image, task, (mapping, prompt))
    return None
