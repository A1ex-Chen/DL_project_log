def b642str(b64):
    return base64.b64decode(b64).decode('utf-8')
