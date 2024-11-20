def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))
