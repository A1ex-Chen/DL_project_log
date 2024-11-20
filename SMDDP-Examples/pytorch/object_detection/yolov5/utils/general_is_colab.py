def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False
