def is_colab():
    """
    Check if the current script is running inside a Google Colab notebook.

    Returns:
        (bool): True if running inside a Colab notebook, False otherwise.
    """
    return ('COLAB_RELEASE_TAG' in os.environ or 'COLAB_BACKEND_VERSION' in
        os.environ)
