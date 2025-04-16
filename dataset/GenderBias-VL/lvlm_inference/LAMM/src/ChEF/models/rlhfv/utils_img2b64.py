def img2b64(img_path):
    img = Image.open(img_path)
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    base64_str = base64_str.decode('utf-8')
    return base64_str
