def id_to_color(id_string: str) ->str:
    hash_object = hashlib.md5(id_string.encode())
    hash_int = int(hash_object.hexdigest(), 16) % 12
    colors = ['\x1b[91m', '\x1b[92m', '\x1b[93m', '\x1b[94m', '\x1b[95m',
        '\x1b[96m', '\x1b[97m', '\x1b[31m', '\x1b[32m', '\x1b[33m',
        '\x1b[34m', '\x1b[35m']
    return colors[hash_int]
