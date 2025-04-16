def str2b64(str):
    return base64.b64encode(str.encode('utf-8')).decode('utf-8')
