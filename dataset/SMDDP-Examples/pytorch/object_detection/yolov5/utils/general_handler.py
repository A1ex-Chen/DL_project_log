def handler(*args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(e)
